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

# 模擬一些函數
@timing_decorator
def load_data():
    """模擬載入數據"""
    print("載入數據中...")
    time.sleep(1.5)  # 模擬耗時操作
    print("數據載入完成！")

@timing_decorator
def process_data():
    """模擬數據處理"""
    print("處理數據中...")
    time.sleep(2)  # 模擬運算
    print("數據處理完成！")

@timing_decorator
def save_results():
    """模擬保存結果"""
    print("儲存結果中...")
    time.sleep(0.8)
    print("結果儲存完成！")

# 主函數
def main():
    print("=== 程式開始 ===")
    logging.info("=== 程式開始 ===")  # 日誌紀錄開始
    load_data()
    process_data()
    save_results()
    logging.info("=== 程式結束 ===")  # 日誌紀錄結束
    print("=== 程式結束 ===")

# 執行主程式
if __name__ == "__main__":
    main()
