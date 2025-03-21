from config import EmbeddingConfiguration
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
import os
from langchain_postgres.vectorstores import PGVector
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.stores import InMemoryStore
from unstructured.partition.pdf import partition_pdf
import uuid
from src.embedding_helper import ImageSummarizer, Element
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter



class RAGGER:
    def __init__(self, embedder_config: EmbeddingConfiguration):
        self.embedder_config = embedder_config
        self.embedder = None
        self.vector_store = None
        self.retriever = None
        self.id_key = 'wifi_doc_id'

        self.embedder, self.vector_store = self._load_embeddings_and_database()
        self.retriever = self._load_retriever()
        self.text_splitter = self._load_text_splitter()


    def _load_embeddings_and_database(self) -> PGVector:
        try:
            if self.embedder_config.embedding_provider == 'openai':
                embedder = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")
            else:
                ollama_embedding_model = 'quentinz/bge-large-zh-v1.5:latest'
                embedder = OllamaEmbeddings(model=ollama_embedding_model, base_url="http://10.96.196.63:11434")

            vector_store = PGVector(
                embeddings=embedder,
                collection_name='WIFI_802.11_Knowledge_Base', # 區分主題(大房間)
                connection=os.getenv('POSTGRES_URL'),
                collection_metadata={"id_field": self.id_key},
            )
            return embedder, vector_store
        except Exception as e:
            raise Exception(f"Error loading embeddings and database: {str(e)}")
        

    def _load_retriever(self) -> MultiVectorRetriever:
        retriever = MultiVectorRetriever(
            vectorstore=self.vector_store,
            docstore=InMemoryStore(),
            id_key=self.id_key, # 區分原始文檔: 告知retriever將透過此key來標示每個子文檔來自哪份原始文檔
        )
        return retriever
    

    def analyze_pdf_component(self, file_path):
        images_path = './images'
        raw_pdf_elements = partition_pdf(
            # filename="weekly_market_recap.pdf",
            # filename="statement_of_changes.pdf",
            filename=f"{file_path}",
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Figure"],
            infer_table_structure=True,
            include_metadata=True,
            include_page_breaks=True,
            # chunking_strategy="by_title",
            max_characters=1500,         # 降低每個段落的最大字元數
            new_after_n_chars=1400,      # 降低觸發新段落的字元數
            combine_text_under_n_chars=500,  # 降低合併門檻，使分段更細
            extract_image_block_output_dir=images_path,
        )
        return raw_pdf_elements
    
    def embed_pdf_images_and_store(self, remote_file, images_path='./images'):
        image_data_list = []
        image_summary_list = []

        for img_file in sorted(os.listdir(images_path)):
            if img_file.endswith(".jpg"):
                summarizer = ImageSummarizer(os.path.join(images_path, img_file))
                data, summary = summarizer.summarize()
                image_data_list.append(data)
                image_summary_list.append(summary)

        if len(image_summary_list) > 0:
            # Add images to vector store
            image_id = [str(uuid.uuid4()) for _ in image_data_list]
            summary_images = [
                Document(page_content=s, metadata={self.id_key: image_id[i], 'id': image_id[i], 'remote_file': remote_file})
                for i, s in enumerate(image_summary_list)
            ]
            self.retriever.vectorstore.add_documents(summary_images)
            self.retriever.docstore.mset(list(zip(image_id, image_data_list)))

    
    def embed_table_text_and_store(self, remote_file, raw_pdf_elements):
        table_elements = []
        text_elements = []
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                table_elements.append(Element(type="table", text=str(element)))
            elif element.category not in ("Image", "PageBreak"):
                text_elements.append(Element(type="text", text=str(element)))


        prompt_text = """
            You are responsible for concisely summarizing table or text chunk:
            {element}
            請使用繁體中文
        """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x} | prompt | ChatOpenAI(temperature=0, model="gpt-4o-mini") | StrOutputParser()


        if len(table_elements) > 0:
            # 1. 處理table文字
            tables = [i.text for i in table_elements]
            table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
            # Add tables
            table_ids = [str(uuid.uuid4()) for _ in tables]
            summary_tables = [
                Document(page_content=s, metadata={self.id_key: table_ids[i], 'id': table_ids[i], 'remote_file': remote_file})
                for i, s in enumerate(table_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_tables)
            self.retriever.docstore.mset(list(zip(table_ids, tables)))


        if len(text_elements) > 0:
            # 2. 處理純文字
            texts = [i.text for i in text_elements]
            original_full_text = " ".join(texts) # Combine all raw text elements
            original_chunks = self.text_splitter.create_documents([original_full_text])
            text_summaries = summarize_chain.batch(original_chunks, {"max_concurrency": 5})

            # Add texts
            doc_ids = [str(uuid.uuid4()) for _ in original_chunks]
            summary_texts = [
                Document(page_content=s, metadata={self.id_key: doc_ids[i], 'id': doc_ids[i], 'remote_file': remote_file})
                for i, s in enumerate(text_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_texts)
            self.retriever.docstore.mset(list(zip(doc_ids, original_chunks)))


    def _load_text_splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                "。",
                ",",
                "!",
                "！",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ],
        )