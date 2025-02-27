from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from config import LLM_PROVIDER, OPENAI_API_KEY
from postgresql import get_pg_engine
from sqlalchemy import text, create_engine
from config import POSTGRES_URL
import psycopg2


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


def get_embedding_model(provider):
    """
    根據 provider 參數選擇要使用的 embedding 模型。
    預設使用 Ollama，但可以透過環境變數或參數切換成 OpenAI。
    """
    if provider == "ollama":
        ollama_embedding_model = 'quentinz/bge-large-zh-v1.5:latest' # 'bge-m3:latest'
        return OllamaEmbeddings(model=ollama_embedding_model, base_url="http://10.96.196.63:11434")  # 你可以換成你在 Ollama 內部訓練的 embedding 模型
    

def retrieve_similar_chunks(query, vector_table, top_k=5) -> tuple:
    try:
        conn = psycopg2.connect(
            host='10.96.196.63',
            database='kmsdb',
            user='biguser',
            password='npspo'
        )

        embedding_model = get_embedding_model("ollama")
        query_vector = embedding_model.embed_query(query)

        # 修改查詢以正確轉換查詢向量
        search_query = f"""
            SELECT
                document_name,
                original_text,
                embedding <=> CAST(%s AS vector) AS cosine_distance,
                file_path,
                metadata
            FROM {vector_table}
            ORDER BY cosine_distance ASC
            LIMIT %s;
        """
        with conn.cursor() as cursor:
            cursor.execute(search_query, (query_vector, top_k))
            results = cursor.fetchall()

        # 調試日志
        print(f"Retrieved chunks from {vector_table}: Total of {len(results)} chunks")

        # 返回相關分塊內容和相似度
        return results

    except Exception as e:
        return ()  # 確保即使出錯也返回空列表，避免後端崩潰
    finally:
        if conn:
            conn.close()



def main():
    query = '請問甚麼是802.11ax'
    results = retrieve_similar_chunks(query, "wifi_knowledge_embedding_bge", top_k=5)


    for result in results:
        print(result)
        print('--------------------------------------')



main()



messages = [
    (
        "system",
        "You are a helpful assistant that translates English to 繁體中文. Translate the user sentence.",
    ),
    ("human", user_inputs),
    ("assistant", "翻譯成繁體中文"),
]


# 呼叫模型並逐步輸出
llm = get_llm()
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)  # 逐字輸出