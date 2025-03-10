import logging
import os
import psutil
from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
import torch

# 設定 logging
logging.basicConfig(
    filename="pdf_parsing.log",  # 記錄到檔案
    level=logging.DEBUG,         # 記錄 DEBUG 以上的訊息
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 檢測GPU可用性
def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA版本: {torch.version.cuda}")
        return device
    else:
        device = torch.device("cpu")
        logging.info("無GPU可用，將使用CPU")
        return device

# 監控記憶體使用
def log_memory_usage():
    # CPU 記憶體
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"CPU記憶體使用: {mem_info.rss / 1024 / 1024:.2f} MB")
    
    # GPU 記憶體
    if torch.cuda.is_available():
        logging.info(f"GPU記憶體已分配: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")
        logging.info(f"GPU記憶體快取: {torch.cuda.memory_reserved()/1024/1024:.2f} MB")

# 監控程式開始
logging.info("=" * 50)
logging.info("開始解析 PDF 文件")
device = check_gpu()
log_memory_usage()

# 設定環境變數以使用GPU (對某些底層庫可能有用)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一個GPU

try:
    # 記錄PDF開始處理
    logging.info("開始處理 PDF: statement_of_changes.pdf")
    log_memory_usage()
    
    # 設定模型優先使用GPU (如果相關模型支援)
    # 注意: 不是所有unstructured的功能都支援GPU
    os.environ["PYTORCH_DEVICE"] = str(device)
    
    # 嘗試分批處理以減少記憶體使用
    raw_pdf_elements = partition_pdf(
        filename="statement_of_changes.pdf",
        extract_images_in_pdf=False,  # 減少記憶體使用
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=".",
        # GPU相關設定 (如果底層函數支援)
        # device=str(device),  # 取消註釋如果partition_pdf支援此參數
    )

    # 記錄處理完成
    log_memory_usage()
    logging.info(f"成功解析 {len(raw_pdf_elements)} 個元素")
    print(f"成功解析 {len(raw_pdf_elements)} 個元素")

    # 儲存結果到檔案
    with open("pdf_elements.txt", "w", encoding="utf-8") as f:
        for i, element in enumerate(raw_pdf_elements):
            f.write(f"元素 {i}:\n{str(element)}\n{'='*50}\n")
    
    # 只在控制台顯示前5個元素
    for i, element in enumerate(raw_pdf_elements[:5]):
        logging.debug(f"元素 {i}: {element}")
        print(f"元素 {i}: {element}")

except Exception as e:
    logging.error(f"解析 PDF 時發生錯誤: {e}", exc_info=True)
    print(f"解析 PDF 時發生錯誤: {e}")
finally:
    # 程式結束時的記憶體狀態
    log_memory_usage()
    logging.info("PDF 處理程序結束")
    
    # 清理GPU記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("已清理GPU記憶體")