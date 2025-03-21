from langchain_postgres.vectorstores import PGVector
from dotenv import load_dotenv
import os
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore


load_dotenv(override=True)



class PostgresVectorDB:

    def __init__(self):
        self.vector_store = self._get_vector_store()


    def _get_vector_store(self, embedder):
        vector_store = PGVector(
            embeddings=embedder,
            collection_name='WIFI_802.11_Knowledge_Base',
            connection=os.getenv('POSTGRES_URL'),
        )
        return vector_store
    

    def get_retriever(self):
        retriever = MultiVectorRetriever(
            vectorstore=self.vector_store,
            docstore=InMemoryStore(),
            id_key="文檔A",
        )
        return retriever