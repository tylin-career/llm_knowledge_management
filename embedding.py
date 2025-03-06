import os
from config import POSTGRES_URL, EMBEDDING_PROVIDER, OPENAI_API_KEY
from postgresql import get_pg_engine
import paramiko
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_core.vectorstores import InMemoryVectorStore
from datetime import datetime
import pytz
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
# from tabulate import tabulate

import json
import sqlalchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

tz = pytz.timezone("Asia/Taipei")



# SSH 連線資訊（請填入你的遠端伺服器資訊）
SSH_HOST = "10.96.196.74"
SSH_PORT = 22  # 預設為 22
SSH_USER = "biguser"
SSH_PASSWORD = "npspo"  # 建議使用 SSH 金鑰驗證，避免明文密碼
BASE_PATH = "/mnt/nfs_share/pydio/jacky/05_Technical_Knowledge/00_Internal_Training/03_WiFi_Professsional"


def ssh_connect():
    """建立 SSH 連線"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD)
    return client

def get_all_files():
    """透過 SSH 取得所有符合指定格式的檔案（排除 test 相關資料夾），並回傳 document_name, file_path, file_ext"""
    client = ssh_connect()
    cmd = f"find {BASE_PATH} -type f | grep -v test"
    stdin, stdout, stderr = client.exec_command(cmd)
    
    files = stdout.read().decode().splitlines()
    client.close()

    # 過濾出符合副檔名的檔案
    valid_extensions = (".docx", ".doc", ".pdf", ".txt", ".md")
    filtered_files = [
        {
            "document_name": os.path.basename(f),  # 取得檔名
            "file_path": f,  # 完整路徑
            "file_ext": os.path.splitext(f)[1]  # 副檔名
        }
        for f in files if f.lower().endswith(valid_extensions)
    ]

    return filtered_files

def download_file_via_ssh(remote_path):
    """
    透過 SFTP 下載遠端檔案到本機
    :param remote_path: 遠端檔案完整路徑
    :return: 本機檔案存放位置
    """

    LOCAL_DOWNLOAD_DIR = "./downloads"
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
        sftp.get(remote_path, local_path)  # 執行檔案下載
        print(f"下載完成: {local_path}")
        
        sftp.close()
        client.close()
        
        return local_path  # 回傳本機檔案路徑
    except Exception as e:
        print(f"下載失敗 {remote_path}: {e}")
        return None

def get_embedding_model(provider):
    """
    根據 provider 參數選擇要使用的 embedding 模型。
    預設使用 Ollama，但可以透過環境變數或參數切換成 OpenAI。
    """
    if provider == "openai":
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")  # 你可以換成其他 OpenAI embedding 模型
    else:
        ollama_embedding_model = 'quentinz/bge-large-zh-v1.5:latest' # 'bge-m3:latest'
        return OllamaEmbeddings(model=ollama_embedding_model, base_url="http://10.96.196.63:11434")  # 你可以換成你在 Ollama 內部訓練的 embedding 模型



def get_loader(local_file_path, file_ext:str):
    if file_ext.lower() in ('.doc','.docx'):
        return UnstructuredWordDocumentLoader(local_file_path)
    if file_ext.lower() == '.pdf':
        return PyPDFLoader(local_file_path)
    if file_ext.lower() == '.txt':
        return TextLoader(local_file_path, encoding='utf-8')
    elif file_ext == '.txt':
        return None



def filter_chunks(chunks, percentile_threshold=30):
    """
    根據 TF-IDF 計算 chunk 的重要性，並過濾低價值 chunk。
    
    參數：
    - chunks: list[str]，要評估的文本片段
    - percentile_threshold: int，設定多少百分位以下的 TF-IDF 分數要過濾
    
    回傳：
    - DataFrame，包含 chunk、TF-IDF 分數、是否保留
    """
    # 計算 TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)
    
    # 計算每個 chunk 的平均 TF-IDF 分數
    tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=1)

    # 設定過濾閾值（低於此閾值的 chunk 會被過濾）
    threshold = np.percentile(tfidf_scores, percentile_threshold)

    # 保留 threshold 以內的 chunk
    filtered_chunks = [(chunk, score, score >= threshold) for chunk, score in zip(chunks, tfidf_scores)]
    
    df = pd.DataFrame(filtered_chunks, columns=["Chunk", "TF-IDF Score", "Keep"])
    # print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
    return df



data = []
files = get_all_files() #[0:1]
print(f'共有 {len(get_all_files())}個檔案')
# files = [{
#     'document_name':'testing_file.docx',
#     'file_path':'./downloads/testing_file.docx',
#     'file_ext':'.docx'
# }]
for file in files:
    current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    document_name = file['document_name']
    file_path = file['file_path']
    file_ext = file['file_ext']

    download_file_via_ssh(file['file_path'])

    loader = get_loader(f'./downloads/{document_name}', file_ext)

    document_text = loader.load()
    document_text_page_content = document_text[0].page_content

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            "。",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
    )
    splitted_chunks = text_splitter.split_text(document_text_page_content)

# 需要過濾掉一些不重要的 chunks，以及預處理 chunks

    df = filter_chunks(splitted_chunks, percentile_threshold=30)
    
    embedding_model = get_embedding_model(EMBEDDING_PROVIDER)

    vectors = embedding_model.embed_documents(splitted_chunks)
    
    for idx, (chunk, vector) in enumerate(zip(splitted_chunks, vectors), start=1):
        data.append([document_name, idx, chunk, '', vector, current_time, file_path, file])


df = pd.DataFrame(data, columns=['document_name', 'chunk_id', 'original_text', 'cleaned_text', 'embedding', 'process_datetime', 'file_path', 'metadata'])
df['cleaned_text'] = df['original_text'].apply(lambda x: x.replace("\n", " ").replace("\r", " ") if isinstance(x, str) else x)
df['metadata'] = df['metadata'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)


# from IPython.display import display
print(df.shape)
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

print("數據插入或更新完成！")