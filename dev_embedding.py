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



import os
import platform
import subprocess
from pathlib import Path


# 設定時區
tz = pytz.timezone("Asia/Taipei")

# SSH 連線資訊
SSH_HOST = "10.96.196.74"
SSH_PORT = 22
SSH_USER = "biguser"
SSH_PASSWORD = "npspo"
BASE_PATH = "/mnt/nfs_share/pydio/jacky/05_Technical_Knowledge/00_Internal_Training/03_WiFi_Professsional"

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
    stdin, stdout, stderr = client.exec_command(cmd)
    
    files = stdout.read().decode().splitlines()
    client.close()

    # 過濾出符合副檔名的檔案
    valid_extensions = (".docx", ".doc", ".pdf", ".txt", ".md")
    filtered_files = [
        {
            "document_name": os.path.basename(f),
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


def get_embedding_model(provider):
    """根據 provider 選擇要使用的 embedding 模型"""
    if provider == "openai":
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    else:
        ollama_embedding_model = 'quentinz/bge-large-zh-v1.5:latest'
        return OllamaEmbeddings(model=ollama_embedding_model, base_url="http://10.96.196.63:11434")

from unstructured.partition.pdf import partition_pdf
def get_loader(local_file_path):

    file_path = Path(f"{local_file_path}")
    print("副檔名：", file_path.suffix)

    """根據檔案類型取得對應的文件載入器"""
    if file_path.suffix in ('.txt'):
        return TextLoader(local_file_path, encoding='utf-8')
    else:
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



def is_ubuntu():
    """檢查系統是否為 Ubuntu"""
    if platform.system() == "Linux":
        try:
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if "Ubuntu" in line:
                        return True
        except Exception as e:
            print("讀取 /etc/os-release 時發生錯誤:", e)
    return False


def main():
    # 取得遠端檔案清單
    remote_files = get_all_files()[0:5]
    print(f'共有 {len(remote_files)}個檔案')

    pdf_doc_list = []
    for remote_file in remote_files:
        document_name, remote_file_path, remote_file_ext = map(
            remote_file.get, ["document_name", "remote_file_path", "remote_file_ext"]
        )
        local_file_path = download_file_via_ssh(remote_file_path) # 存到本地
        # local_file_path = /home/biguser/codeFactory/llm_knowledge_management/downloads/Wi-Fi 6(802.11ax)解析3：上行随机接入（TF，TF-R）.pdf


        if remote_file_ext in {".docx", ".doc"}:
            # 將 docx 轉換成 pdf
            local_file_path = convert_docx_to_pdf(local_file_path, LOCAL_DOWNLOAD_DIR)

        # 如果是純文字檔案，如 txt
        if local_file_path.endswith(".txt"):
            loader = get_loader(local_file_path)
            document_text = loader.load()
            print(f'{local_file_path}準備text_split並存入vector store')
        else:
            print(f'{local_file_path}開始處理 PDF')



        print('----------------')
    # print(len(pdf_doc_list))
    # print((pdf_doc_list))


if __name__ == "__main__":
    main()