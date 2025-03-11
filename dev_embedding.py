import os
import re
import json
import numpy as np
import pandas as pd
import paramiko
import pytz
import sqlalchemy
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

# Langchain 相關引入
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 從 config 引入相關設定
from config import POSTGRES_URL, EMBEDDING_PROVIDER, OPENAI_API_KEY
from postgresql import get_pg_engine

# 設定時區
tz = pytz.timezone("Asia/Taipei")

# SSH 連線資訊
SSH_HOST = "10.96.196.74"
SSH_PORT = 22
SSH_USER = "biguser"
SSH_PASSWORD = "npspo"
BASE_PATH = "/mnt/nfs_share/pydio/jacky/05_Technical_Knowledge/00_Internal_Training/03_WiFi_Professsional"

# 設定下載資料夾
LOCAL_DOWNLOAD_DIR = "./downloads"


def ssh_connect():
    """建立 SSH 連線"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD)
    return client


def get_all_files():
    """透過 SSH 取得所有符合指定格式的檔案（排除 test 相關資料夾）"""
    client = ssh_connect()
    cmd = f"find {BASE_PATH} -type f | grep -v test"
    stdin, stdout, stderr = client.exec_command(cmd)
    
    files = stdout.read().decode().splitlines()
    client.close()

    # 過濾出符合副檔名的檔案
    valid_extensions = (".docx", ".doc", ".pdf", ".txt", ".md")
    filtered_files = [
        {
            "document_name": os.path.basename(f),
            "file_path": f,
            "file_ext": os.path.splitext(f)[1].lower()
        }
        for f in files if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    return filtered_files


def download_file_via_ssh(remote_path):
    """透過 SFTP 下載遠端檔案到本機"""
    # 確保本機下載資料夾存在
    if not os.path.exists(LOCAL_DOWNLOAD_DIR):
        os.makedirs(LOCAL_DOWNLOAD_DIR)

    # 取得遠端檔案名稱
    file_name = os.path.basename(remote_path)
    local_path = os.path.join(LOCAL_DOWNLOAD_DIR, file_name)

    try:
        client = ssh_connect()
        sftp = client.open_sftp()
        
        print(f"正在下載: {remote_path} -> {local_path}")
        sftp.get(remote_path, local_path)
        print(f"下載完成: {local_path}")
        
        sftp.close()
        client.close()
        
        return local_path
    except Exception as e:
        print(f"下載失敗 {remote_path}: {e}")
        return None


def get_embedding_model(provider):
    """根據 provider 選擇要使用的 embedding 模型"""
    if provider == "openai":
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    else:
        ollama_embedding_model = 'quentinz/bge-large-zh-v1.5:latest'
        return OllamaEmbeddings(model=ollama_embedding_model, base_url="http://10.96.196.63:11434")


def get_loader(local_file_path, file_ext):
    """根據檔案類型取得對應的文件載入器"""
    if file_ext in ('.doc', '.docx'):
        return UnstructuredWordDocumentLoader(local_file_path)
    elif file_ext == '.pdf':
        return PyPDFLoader(local_file_path)
    elif file_ext in ('.txt', '.md'):
        return TextLoader(local_file_path, encoding='utf-8')
    else:
        return None


def preprocess_text(text):
    """預處理文本，使切分更有效"""
    # 替換多餘的空白和特殊字元
    text = text.replace("\u200b", "")  # 移除零寬空格
    text = re.sub(r'\s+', ' ', text)  # 合併多個空白
    
    # 確保句子和段落邊界清晰
    text = re.sub(r'([。！？!?])\s*', r'\1\n', text)  # 在句末添加換行
    
    # 修復可能的錯誤分隔，例如 "802.11" 這樣的技術標準不應該在點後換行
    text = re.sub(r'(\d+)\.(\d+)\n', r'\1.\2 ', text)
    
    return text


def split_text_intelligently(text, min_chunk_size=300, max_chunk_size=800, min_overlap=100):
    """智能切分文本，儘量維持語意完整性"""
    # 定義分隔符號，從較大的語意單元到較小的
    separators = [
        "\n\n\n",  # 章節分隔
        "\n\n",    # 段落分隔
        "\n",      # 行分隔
        "。", "！", "？",  # 中文句號
        ".", "!", "?",    # 英文句號
        "；", ";",        # 分號
        "，", ",", "、",  # 逗號
        " ",              # 空格
        ""                # 字元級別
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=min_overlap,
        length_function=len,
        separators=separators
    )
    
    # 第一步切分
    chunks = text_splitter.split_text(text)
    
    # 修正非常短的片段
    merged_chunks = []
    current_chunk = ""
    
    for chunk in chunks:
        if len(current_chunk) + len(chunk) < max_chunk_size:
            current_chunk += " " + chunk if current_chunk else chunk
        else:
            if current_chunk:
                merged_chunks.append(current_chunk)
            current_chunk = chunk
    
    if current_chunk:
        merged_chunks.append(current_chunk)
    
    # 確保每個 chunk 的完整性
    refined_chunks = []
    for chunk in merged_chunks:
        chunk = chunk.strip()
        
        # 太短的片段可能不太有用
        if len(chunk) < min_chunk_size:
            continue
        
        # 檢查是否以句子中間開始
        first_sentence_end = max(
            chunk.find('。'), chunk.find('！'), chunk.find('？'),
            chunk.find('.'), chunk.find('!'), chunk.find('?')
        )
        
        # 如果第一個句子太短，可能是斷在中間
        if 0 < first_sentence_end < 20:
            # 尋找下一個句子結束
            next_end = max(
                chunk.find('。', first_sentence_end + 1),
                chunk.find('！', first_sentence_end + 1),
                chunk.find('？', first_sentence_end + 1),
                chunk.find('.', first_sentence_end + 1),
                chunk.find('!', first_sentence_end + 1),
                chunk.find('?', first_sentence_end + 1)
            )
            if next_end > 0:
                chunk = chunk[first_sentence_end + 1:].strip()
        
        # 檢查是否斷在句子中間
        last_period = max(
            chunk.rfind('。'), chunk.rfind('！'), chunk.rfind('？'),
            chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?')
        )
        
        if last_period > 0 and last_period < len(chunk) - 1:
            # 如果句子結束點在後面的部分，保留到該處
            if last_period > len(chunk) * 0.7:
                chunk = chunk[:last_period + 1].strip()
        
        refined_chunks.append(chunk)
    
    return refined_chunks


def evaluate_chunk_quality(chunks):
    """評估和過濾 chunk 品質"""
    # 使用 TF-IDF 計算文本重要性
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # 嘗試 fit_transform，處理可能的空值
    valid_chunks = [c for c in chunks if isinstance(c, str) and len(c.strip()) > 0]
    if not valid_chunks:
        return pd.DataFrame(columns=["Chunk", "TF-IDF Score", "Completeness", "Keep"])
    
    tfidf_matrix = vectorizer.fit_transform(valid_chunks)
    tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=1)
    
    # 計算完整性分數 (基於句子結束符號)
    completeness_scores = []
    for chunk in valid_chunks:
        # 檢查是否以完整句子開始和結束
        starts_complete = bool(re.match(r'^[A-Z\u4e00-\u9fff「『（《【]', chunk))
        ends_complete = bool(re.search(r'[.。!！?？"」』）》】]$', chunk))
        
        # 計算包含多少個完整句子
        sentence_endings = len(re.findall(r'[.。!！?？]', chunk))
        
        # 結合以上因素
        completeness = (0.3 * starts_complete + 0.3 * ends_complete + 
                        0.4 * min(1.0, sentence_endings / 3))
        completeness_scores.append(completeness)
    
    # 設定分數閾值
    tfidf_threshold = np.percentile(tfidf_scores, 20)  # 保留前 80%
    completeness_threshold = 0.5
    
    # 綜合評估保留哪些 chunk
    evaluation = []
    for chunk, tfidf, completeness in zip(valid_chunks, tfidf_scores, completeness_scores):
        keep = (tfidf >= tfidf_threshold and completeness >= completeness_threshold)
        evaluation.append((chunk, tfidf, completeness, keep))
    
    return pd.DataFrame(evaluation, columns=["Chunk", "TF-IDF Score", "Completeness", "Keep"])


def main():
    # 初始化資料列表
    data = []
    files = get_all_files()
    print(f'共有 {len(files)} 個檔案')
    
    for file in files:
        try:
            current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
            document_name = file['document_name']
            file_path = file['file_path']
            file_ext = file['file_ext']
            
            # 下載檔案
            local_path = download_file_via_ssh(file_path)
            if not local_path:
                print(f"跳過處理 {document_name}，下載失敗")
                continue
            
            # 載入文件
            loader = get_loader(local_path, file_ext)
            if not loader:
                print(f"跳過處理 {document_name}，不支援的檔案類型: {file_ext}")
                continue
            
            # 讀取內容
            try:
                document = loader.load()
                if not document or not document[0].page_content:
                    print(f"跳過處理 {document_name}，文件內容為空")
                    continue
                
                document_text = document[0].page_content
            except Exception as e:
                print(f"讀取 {document_name} 失敗: {e}")
                continue
            
            # 預處理文本
            preprocessed_text = preprocess_text(document_text)
            
            # 智能切分文本
            chunks = split_text_intelligently(preprocessed_text)
            print(f"{document_name}: 初步切分為 {len(chunks)} 個片段")
            
            # 評估和過濾 chunks
            evaluation = evaluate_chunk_quality(chunks)
            quality_chunks = evaluation[evaluation["Keep"]]["Chunk"].tolist()
            print(f"{document_name}: 過濾後保留 {len(quality_chunks)} 個高品質片段")
            
            # 進行 Embedding
            embedding_model = get_embedding_model(EMBEDDING_PROVIDER)
            vectors = embedding_model.embed_documents(quality_chunks)
            
            # 將資料加入列表
            for idx, (chunk, vector) in enumerate(zip(quality_chunks, vectors), start=1):
                chunk_data = {
                    'document_name': document_name,
                    'chunk_id': f"{document_name.split('.')[0]}_{idx}",
                    'original_text': chunk,
                    'cleaned_text': chunk.replace("\n", " ").strip(),
                    'embedding': vector,
                    'process_datetime': current_time,
                    'file_path': file_path,
                    'metadata': file
                }
                data.append(chunk_data)
            
            print(f"完成處理 {document_name}")
            
        except Exception as e:
            print(f"處理 {file['document_name']} 時發生錯誤: {e}")
    
    # 轉換為 DataFrame
    if not data:
        print("沒有資料可處理，程式結束")
        return
    
    df = pd.DataFrame(data)
    print(f"總共處理了 {df.shape[0]} 個文本片段")
    
    # 準備 metadata 欄位
    df['metadata'] = df['metadata'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
    
    # 儲存到 PostgreSQL
    try:
        pg_engine = get_pg_engine()
        dtype_schema = {
            'document_name': sqlalchemy.types.Unicode,
            'chunk_id': sqlalchemy.types.String,
            'original_text': sqlalchemy.types.Unicode,
            'cleaned_text': sqlalchemy.types.Unicode,
            'embedding': Vector(1024),
            'process_datetime': sqlalchemy.types.DateTime,
            'file_path': sqlalchemy.types.String,
            'metadata': JSONB
        }
        df.to_sql("wifi_knowledge_embedding_bge", pg_engine, dtype=dtype_schema, if_exists="replace", index=False)
        print("資料庫儲存成功！共插入 {} 筆資料".format(len(df)))
    except Exception as e:
        print(f"儲存到資料庫時發生錯誤: {e}")


if __name__ == "__main__":
    main()