import time
import functools
import logging

# 設定 logging，寫入日誌檔案
logging.basicConfig(
    filename="execution_log.log",  # 日誌文件名稱
    level=logging.INFO,            # 記錄 INFO 級別以上的訊息
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 計時 Decorator，並記錄到日誌
def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        log_message = f"函數 {func.__name__} 執行時間: {execution_time:.6f} 秒"
        
        # 終端機輸出
        print(log_message)
        
        # 寫入日誌
        logging.info(log_message)
        
        return result
    return wrapper
