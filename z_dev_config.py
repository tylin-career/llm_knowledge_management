from dotenv import load_dotenv
import os
import getpass
import json
from dataclasses import dataclass, field


load_dotenv(override=True)

@dataclass
class Configuration:
    mode: str = ['PROD','DEV'][0]
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY') or getpass.getpass('Enter your OpenAI API key: ')
    POSTGRES_USER: str = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD: str = os.getenv('POSTGRES_PASSWORD')
    POSTGRES_DB: str = os.getenv('POSTGRES_DB')
    POSTGRES_HOST: str = os.getenv('POSTGRES_HOST')
    POSTGRES_PORT: str = os.getenv('POSTGRES_PORT')
    POSTGRES_URL: str = os.getenv('POSTGRES_URL')

    # 用於儲存 JSON 設定的屬性
    LLM_PROVIDER: str = field(init=False)
    EMBEDDING_PROVIDER: str = field(init=False)

    def __post_init__(self):
        """在 `__init__` 之後執行，讀取 JSON 設定檔"""
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)

            # 儲存 JSON 設定到物件內
            self.LLM_PROVIDER = config.get('LLM_PROVIDER', 'openai')
            self.EMBEDDING_PROVIDER = config.get('EMBEDDING_PROVIDER', 'openai')

        except FileNotFoundError:
            self.LLM_PROVIDER = "openai"
            self.EMBEDDING_PROVIDER = "openai"
            print("⚠️ config.json 不存在，使用預設值")

# 測試初始化
config = Configuration()
print(f"Mode: {config.mode}")
print(f"LLM Provider: {config.LLM_PROVIDER}")
print(f"Embedding Provider: {config.EMBEDDING_PROVIDER}")