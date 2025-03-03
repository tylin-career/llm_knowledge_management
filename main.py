from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from config import LLM_PROVIDER, OPENAI_API_KEY
from postgresql import get_pg_engine
from config import POSTGRES_URL
import psycopg2
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory


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
    if provider == "openai":
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")  # 你可以換成其他 OpenAI embedding 模型
    else:
        ollama_embedding_model = 'quentinz/bge-large-zh-v1.5:latest' # 'bge-m3:latest'
        return OllamaEmbeddings(model=ollama_embedding_model, base_url="http://10.96.196.63:11434")  # 你可以換成你在 Ollama 內部訓練的 embedding 模型
    

def retrieve_similar_chunks(query, vector_table, top_k=5) -> list:
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
                file_path
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
        return []  # 確保即使出錯也返回空列表，避免後端崩潰
    finally:
        if conn:
            conn.close()


def main():

    # https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html
    prompt = ChatPromptTemplate(
            [
                ("system", """
                你是一位在 WiFi 6、WiFi 7 與 802.11 協議的專家，請根據參考資訊回答問題：
                
                參考資訊：
                {context}

                注意：
                1. 你只能依據提供的資訊回答，請勿編造內容。
                2. 若無足夠資訊，請回答「根據目前資訊無法回答」。
                3. 請以專業、精確的方式，以繁體中文為主回答問題。
                4. 如果是打招呼，請禮貌回復。
                """),
                ("human", '{user_query}'), 
                ("ai", """
                    您的問題是: {user_query}。
                    我將參考 {context} 回答您的問題。
                """), 
                MessagesPlaceholder("conversation")  # 用來放對話歷史
            ]
        )

    llm_model = get_llm()

    # **3. 啟用對話記憶**
    memory = ConversationBufferMemory(memory_key="conversation", return_messages=True)

    # **6. 建立串流 chain**
    chain = prompt | llm_model
    # chain = prompt | llm | StrOutputParser()

    # **7. 問答迴圈**
    while True:
        user_query = input("請輸入你的問題（輸入 'exit' 離開）：")
        if user_query.lower() == "exit":
            break
        
        # **更新記憶**
        memory.chat_memory.add_user_message(user_query)



        #### ---------------Retrieve similar chunks---------------
        retrieved_data = retrieve_similar_chunks(user_query, "wifi_knowledge_embedding_bge", top_k=5)

        # Get retrieved context and its similiarity
        context_list = list(zip([context[1] for context in retrieved_data], [context[2] for context in retrieved_data]))
        # Get file_name and its remote path
        file_info_list = list(zip([document[0] for document in retrieved_data], [document[3] for document in retrieved_data]))

        context_chunks = [thing[0] for thing in context_list]
    ## RERANKING
        formatted_context = "\n\n".join(context_chunks)
        #### ---------------Retrieve similar chunks---------------



        print("\nAI 回答：", end="", flush=True)
        full_conversation = ''

        for ai_reply in chain.stream({
            "context": formatted_context,
            "user_query": user_query,
            "conversation": memory.load_memory_variables({})["conversation"]
        }):
            print(ai_reply.content, end="", flush=True)
            full_conversation += ai_reply.content

        print("\n")  # 換行

        # **儲存 AI 回應**
        memory.chat_memory.add_ai_message(full_conversation)

if __name__ == "__main__":
    main()
