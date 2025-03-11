from dotenv import load_dotenv
import os
import getpass


load_dotenv()

class Configuration:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or getpass.getpass('Enter your OpenAI API key: ')
        self.LLM_PROVIDER = os.getenv('LLM_PROVIDER')
        self.POSTGRES_URL = os.getenv('POSTGRES_URL')
        self.EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER')