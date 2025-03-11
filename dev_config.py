from dotenv import load_dotenv
import os
import getpass
import json


load_dotenv()


class Configuration:
    def __init__(self):
        self.mode = ['PROD','DEV'][0]
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or getpass.getpass('Enter your OpenAI API key: ')
        self.POSTGRES_USER = os.getenv('POSTGRES_USER')
        self.POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
        self.POSTGRES_DB = os.getenv('POSTGRES_DB')
        self.POSTGRES_HOST = os.getenv('POSTGRES_HOST')
        self.POSTGRES_PORT = os.getenv('POSTGRES_PORT')
        self.POSTGRES_URL = os.getenv('POSTGRES_URL')


        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        self.LLM_PROVIDER = config.get('LLM_PROVIDER')
        self.EMBEDDING_PROVIDER = config.get('EMBEDDING_PROVIDER')