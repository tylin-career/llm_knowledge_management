from dotenv import load_dotenv
import os
import getpass


load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or getpass.getpass('Enter your OpenAI API key: ')
LLM_PROVIDER = os.getenv('LLM_PROVIDER')
POSTGRES_URL = os.getenv('POSTGRES_URL')
EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER')
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')

os.environ["LANGSMITH_TRACING"] = "true"