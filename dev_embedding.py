import os
import subprocess
from pathlib import Path
import shutil
import paramiko
import pytz
from datetime import datetime
from langchain_community.document_loaders import TextLoader
from src.ragger import RAGGER
from config import EmbeddingConfiguration
from timer import timing_decorator
from tqdm import tqdm
import json


# 設定時區
tz = pytz.timezone("Asia/Taipei")

# SSH 連線資訊
with open("config_pydio.json", "r") as f:
    config = json.load(f)
SSH_HOST = config["SSH_HOST"]
SSH_PORT = config["SSH_PORT"]
SSH_USER = config["SSH_USER"]
SSH_PASSWORD = config["SSH_PASSWORD"]
BASE_PATH = config["BASE_PATH"]

# 設定下載資料夾
LOCAL_DOWNLOAD_DIR = os.path.abspath("./downloads")


def ssh_connect():
    """建立 SSH 連線"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD)
    return client


def get_all_files() -> list[dict]:
    """透過 SSH 取得所有符合指定格式的檔案（排除 test 相關資料夾）"""
    client = ssh_connect()
    cmd = f"find {BASE_PATH} -type f | grep -v test"
    _, stdout, _ = client.exec_command(cmd)
    
    files = stdout.read().decode().splitlines()
    client.close()

    # 過濾出符合副檔名的檔案
    valid_extensions = (".docx", ".doc", ".pdf", ".txt", ".md")
    filtered_files = [
        {
            "file_name": os.path.basename(f),
            "remote_file_path": f,
            "remote_file_ext": os.path.splitext(f)[1].lower()
        }
        for f in files if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    return filtered_files


def is_file_updated(sftp, remote_file, local_file):
    """檢查遠端檔案是否需要更新"""
    try:
        remote_size = sftp.stat(remote_file).st_size
        local_size = os.path.getsize(local_file)
        return remote_size != local_size  # 只有大小不同才下載
    except FileNotFoundError:
        return True  # 本地檔案不存在，必須下載


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

        if is_file_updated(sftp, remote_path, local_path):
            print(f"正在下載或更新: {remote_path} -> {local_path}")
            sftp.get(remote_path, local_path)
        else:
            print(f"檔案已存在，無需更新: {local_path}")

        sftp.close()
        client.close()
        
        return local_path
    except Exception as e:
        print(f"下載失敗 {remote_path}: {e}")
        return None


def convert_docx_to_pdf(input_path, output_dir):
    # 執行 LibreOffice 轉換指令
    subprocess.run(
        ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", output_dir, input_path],
        check=True
    )
    # 取得原始檔案的檔名並替換成 .pdf 副檔名
    base_name = os.path.basename(input_path)
    file_name, _ = os.path.splitext(base_name)
    pdf_file = file_name + ".pdf"
    
    # 組合成完整的 PDF 路徑
    output_pdf_path = os.path.join(output_dir, pdf_file)
    return output_pdf_path


def get_loader(local_file_path):

    file_path = Path(f"{local_file_path}")
    print("副檔名：", file_path.suffix)

    """根據檔案類型取得對應的文件載入器"""
    if file_path.suffix in ('.txt'):
        return TextLoader(local_file_path, encoding='utf-8')
    else:
        return None



@timing_decorator
def embedding_and_storing_main(ragger: RAGGER):
    # 取得遠端檔案 清單
    remote_files = get_all_files()
    print(f'共有 {len(remote_files)}個檔案')

    for remote_file in tqdm(remote_files, desc="處理遠端檔案中"):
        file_name, remote_file_path, remote_file_ext = map(
            remote_file.get, ["file_name", "remote_file_path", "remote_file_ext"]
        )
        local_file_path = download_file_via_ssh(remote_file_path) # 存到本地
        # local_file_path = /home/biguser/codeFactory/llm_knowledge_management/downloads/Wi-Fi 6(802.11ax)解析3：上行随机接入（TF，TF-R）.pdf


        if remote_file_ext in {".docx", ".doc"}:
            # 將 docx 轉換成 pdf
            local_file_path = convert_docx_to_pdf(local_file_path, LOCAL_DOWNLOAD_DIR)


        # 如果是純文字檔案，如 txt
        if local_file_path.endswith(".txt"):
            loader = get_loader(local_file_path)
            original_full_text = loader.load()
            print(f'{local_file_path}準備text_split並存入vector store')
            continue
            # original_chunks = ragger.text_splitter.create_documents([original_full_text])
            # text_summaries = summarize_chain.batch(original_chunks, {"max_concurrency": 5})
            

        elif local_file_path.endswith(".pdf"):
            print(f'開始處理 pdf 路徑: {local_file_path}')
            raw_pdf_elements = ragger.analyze_pdf_component(local_file_path)

            # 1. 存 images 資料夾的圖片到 vector store
            ragger.embed_pdf_images_and_store(remote_file)
            # 刪除 images 資料夾下的圖片
            shutil.rmtree("./images")
            # 2. 處理文字
            ragger.embed_table_text_and_store(remote_file, raw_pdf_elements)

        else:
            print(f'跳過 {local_file_path}')
            continue


        print(f'完成 {file_name} 的處理')
        print('----------------')


if __name__ == "__main__":
    embedder_config = EmbeddingConfiguration() # Initiate Embedding 專用的 Config 物件
    ragger = RAGGER(embedder_config) # 啟用 RAGGER(pgvector + embedder) 物件
    embedding_and_storing_main(ragger)
    