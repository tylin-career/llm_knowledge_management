from langchain_openai import ChatOpenAI
from config import LLM_PROVIDER, OPENAI_API_KEY
from embedding import get_embedding_model
from postgresql import get_pg_engine
from sqlalchemy import text
from ollama import OllamaEmbeddings


pg_engine = get_pg_engine()

def get_llm():
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=OPENAI_API_KEY,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True,
            # base_url="...",
            # organization="...",
            # other params...
        )
    elif LLM_PROVIDER == "ollama":
        return ChatOpenAI(
            model="llama3.1",
            openai_api_key="ollama",
            openai_api_base="http://10.96.196.63:11434/v1/",
            streaming=True,
        )
    else:
        raise ValueError("不支援的 LLM 提供者，請使用 'ollama' 或 'openai'")


embedding_model = get_embedding_model(provider="ollama")

def get_query_embedding(embedding_model:OllamaEmbeddings):
    user_inputs = input("請輸入詢問句子: ")
    query_embedding = embedding_model.embed_query(user_inputs)
    return query_embedding


def similarity_search(query_embedding, top_k=5):
    search_query = """
        SELECT
            id,
            document_name,
            chunk_id,
            original_text,
            cleaned_text,
            embedding,
            process_datetime,
            file_path,
            metadata
        FROM public."wifi_knowledge_embedding_bge"
        WHERE embedding IS NOT NULL
        ORDER BY embedding <-> :query_embedding
        LIMIT 3
    """        
    # 使用 SQLAlchemy 的 `execute` 來執行 SQL 查詢
    with pg_engine.connect() as conn:
        result = conn.execute(text(search_query), {"query_embedding": query_embedding, "top_k": top_k})
        columns = result.keys()  # 獲取欄位名稱
        #results = result.fetchall()  # 取得查詢結果
        results = [dict(zip(columns, row)) for row in result.fetchall()]    # 拼裝成字典 這樣有欄位資訊比較好餵進去LLM
    return results


query_embedding = get_query_embedding()
results = similarity_search(query_embedding)
print(results)








# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to 繁體中文. Translate the user sentence.",
#     ),
#     ("human", user_inputs)
# ]


# # 呼叫模型並逐步輸出
# llm = get_llm()
# for chunk in llm.stream(messages):
#     print(chunk.content, end="", flush=True)  # 逐字輸出