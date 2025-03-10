import logging
from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

# 設定 logging
logging.basicConfig(
    filename="pdf_parsing.log",  # 記錄到檔案
    level=logging.DEBUG,         # 記錄 DEBUG 以上的訊息
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 監控程式開始
logging.info("開始解析 PDF 文件")

try:
    raw_pdf_elements = partition_pdf(
        filename="statement_of_changes.pdf",
        extract_images_in_pdf=False,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=".",
    )

    # 顯示 PDF 解析結果
    logging.info(f"成功解析 {len(raw_pdf_elements)} 個元素")
    print(f"成功解析 {len(raw_pdf_elements)} 個元素")

    for i, element in enumerate(raw_pdf_elements[:5]):  # 只顯示前 5 個元素
        logging.debug(f"元素 {i}: {element}")
        print(f"元素 {i}: {element}")

except Exception as e:
    logging.error(f"解析 PDF 時發生錯誤: {e}", exc_info=True)
    print(f"解析 PDF 時發生錯誤: {e}")
