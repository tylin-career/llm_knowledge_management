from dotenv import load_dotenv
import os
from dataclasses import dataclass


load_dotenv(override=True)

LLM_PROVIDER = os.getenv('LLM_PROVIDER')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
POSTGRES_URL = os.getenv('POSTGRES_URL')
EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER')

class LLMConfiguration:
    def __init__(self, model, openai_api_key, openai_api_base, temperature):
        self.model = model
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base
        self.temperature = temperature


@dataclass
class EmbeddingConfiguration:
    postgresql_url: str = POSTGRES_URL
    embedding_provider: str = EMBEDDING_PROVIDER