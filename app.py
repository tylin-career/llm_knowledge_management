from langchain_openai import ChatOpenAI
import streamlit as st
from config import OPENAI_API_KEY
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config import LLMConfiguration
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from sqlalchemy import create_engine
import sqlalchemy
from langchain_core.runnables import RunnableSequence
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere.rerank import CohereRerank
import time
from langchain.retrievers.multi_vector import MultiVectorRetriever
import os
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.stores import InMemoryStore
import uuid


username = st.session_state.get("username", None)
if not username:
    pass
session_id = str(uuid.uuid4()) 

st.set_page_config(page_title='Streamlit 知識管理對話系統', page_icon='📊', layout='wide', initial_sidebar_state='expanded')
st.title("💬 Chatbot")
st.caption("🚀 ASUS Knowledge Management Simulation Powered by NPSPO")


# 初始化聊天歷史
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "source_documents" not in st.session_state:
    st.session_state["source_documents"] = []


with st.sidebar:
    st.title("Navigation and Settings")
    st.caption("🔧 這是側邊欄的內容")
    model = st.selectbox(
        'Model', ['llama3.1', 'gpt-4o-mini']
    )
    with st.form(key='my_form'):
        if model == "gpt-4o-mini":
            openai_api_base = 'https://api.openai.com/v1/'
            openai_api_key = st.text_input(
                'OpenAI API Key', value = OPENAI_API_KEY, type = 'password'
            )
        else:
            openai_api_base = 'http://10.96.196.63:11434/v1/'
            openai_api_key = 'ollama'
            
        temperature = st.slider(
            'Temperature', 0.0, 1.0, value = 0.6, step = 0.1
        )

        if "config" not in st.session_state:
            st.session_state.config = LLMConfiguration(
                model=model,
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
                temperature=temperature,
            )


        submit_button = st.form_submit_button("儲存設定")
        if submit_button:
            st.session_state.config = LLMConfiguration(
                model=model,
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
                temperature=temperature,
            )
            st.success(f"設定已更新！{model}使用中")



    st.markdown('---')
    if st.sidebar.button('Clear Chat History'):
        st.session_state.clear()
        st.session_state["messages"] = []
        st.rerun()
    st.markdown('---')
    uploaded_file = st.file_uploader("📂 Upload Files", type=["doc", "docx", "txt", "md", "pdf"])




# 顯示歷史聊天記錄
for idx, msg in enumerate(st.session_state.messages):
    if isinstance(msg, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("AI"):
            st.markdown(msg.content)
            # 只在 AI 回應後顯示相關的參考文檔
            if idx // 2 < len(st.session_state.source_documents):  # 每對人機對話對應一個文檔參考
                with st.expander('See Sources'):
                    src_docs = st.session_state.source_documents[idx // 2]
                    if isinstance(src_docs, list) and len(src_docs) > 0 and isinstance(src_docs[0], tuple):
                        for i, (document_name, original_text, cosine_distance) in enumerate(src_docs):
                            st.markdown("**Source:**")
                            file_path = f'./downloads/{document_name}'
                            
                            # 確保檔案存在
                            try:
                                with open(file_path, "rb") as file:
                                    # 按下按鈕時，更新 session_state
                                    if st.download_button(
                                        label=f"📥 {document_name}", 
                                        data=file, 
                                        file_name=document_name, 
                                        key=f"download_{idx}_{i}"
                                    ):
                                        pass  # 按鈕功能已經在 st.download_button 中實現
                            except FileNotFoundError:
                                st.warning(f"檔案 {document_name} 不存在")

                            # Content 換行並加入 Tab 縮排
                            st.markdown("**Content:**  \n" + f"&emsp;&emsp;{original_text}", unsafe_allow_html=True)
                            st.write(
                                f'**Relavance Score：** {round(cosine_distance * 100, 2)}%'
                            )
                            st.divider()



def insert_and_eval_to_001GOD_BUDataLog(df: pd.DataFrame):
    # 定义连接字符串
    conn_str = (
        "mssql+pyodbc://BUData_User:asus$1234@172.22.37.60/BUDataLog"
        "?driver=ODBC+Driver+17+for+SQL+Server&charset=utf8&encoding=utf8"
    )
    engine = create_engine(conn_str)

    dtype = {
        'username': sqlalchemy.types.Unicode,
        'user_query': sqlalchemy.types.Unicode,
        'ai_response': sqlalchemy.types.Unicode,
        'prompt_template': sqlalchemy.types.Unicode,
        'reference_doc': sqlalchemy.types.Unicode,
        'rating_details': sqlalchemy.types.Unicode,
        'note': sqlalchemy.types.Unicode,
        'insert_time': sqlalchemy.types.DateTime,
    }

 
    # 插入数据
    df.to_sql("Knowledge_Query_Log", engine, if_exists="append", index=False, dtype=dtype)


def get_llm(model, openai_api_key, openai_api_base, temperature):
    print(f'使用 {model}')
    if model == "gpt-4o-mini":
        return ChatOpenAI(
            model=model,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True,
        )
    elif model == "llama3.1":
        return ChatOpenAI(
            model=model,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            temperature=temperature,
            streaming=True,
        )

template = """你是一位專精於 WiFi 6、WiFi 7 及 802.11 無線網路協議的技術專家，同時也是知識管理系統中的高階技術講師。以下提供的參考資訊是一段完整的文字，請詳細閱讀並根據該文字回答用戶提問，回答時請僅依據以下文字內容與前文對話紀錄，不得自行延伸。

---
**問題**：
{question}

**參考資訊**：
{context}
---

### **回應要求**：
1. 請先從參考資訊中提取關鍵概念與重要資訊。
2. 若參考資訊無法完全支持回答，請直接回覆：「這個問題似乎與WIFI不相關，請再試一次。」並提供進一步查詢的建議或關鍵詞。
3. 若參考資訊足夠，請按照下列格式回答：
   - **概述**：簡述使用者問題以及回答的核心要點（1-2 句）。
   - **技術回答**：基於摘錄的關鍵資訊，提供具體且清晰的回答。
   - **背景與定義**：必要時補充相關技術定義與背景知識。
   - **條列式總結**：
     - [✓] 核心觀點一
     - [✓] 核心觀點二
     - [✓] 核心觀點三（如有）
   - **進一步學習建議**：附上可能的後續查詢方向或關鍵詞提示以及建議。

     
請確保使用**繁體中文**回答，並保持專業且結構化的語言風格。

    """

def get_response(user_query, formatted_context):

    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm('llama3.1', 'ollama', 'http://10.96.196.63:11434/v1/', 0.6)
    
    # 組合流水線時，確保鍵名稱與模板變數一致，且所有輸入均為字串
    chain = {
        "user_query": RunnablePassthrough(), 
        "formatted_context": RunnablePassthrough()
    } | prompt | llm | StrOutputParser()
    
    return chain.stream({
        'user_query': user_query,
        'formatted_context': formatted_context,
    })


def new_get_response(question, compression_retriever) -> RunnableSequence.stream:

    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm('llama3.1', 'ollama', 'http://10.96.196.63:11434/v1/', 0.6)
    
    # 提示的輸入預期是具有鍵「context」和「question」。使用者只輸入問題。因此，我們需要使用檢索器取得上下文，並在「question」鍵下傳遞使用者輸入。RunnablePassthrough 可讓我們將使用者的問題傳遞到提示和模型。
    chain = {
        "context": compression_retriever, 
        "question": RunnablePassthrough()
    } | prompt | llm | StrOutputParser()
    return chain.stream(question)



def evaluate_response(user_query, ai_response):
    llm = get_llm('gpt-4o-mini', OPENAI_API_KEY, 'https://api.openai.com/v1/', 0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名問答評分專家（QA 評分專家），負責根據準確性、完整性、流暢度等標準，為 AI 的回答進行 1 到 10 的評分。\n"
                "請根據使用者的問題 `{user_query}`，評估 AI 的回應，並給出一個 10 分制的評分。",
            ),
            ("human", "{ai_response}"),
        ]
    )

    chain = prompt | llm
    rating_details = chain.invoke(
        {
            "user_query": user_query,
            "ai_response": ai_response,
        }
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "請根據使用者的問題 `{user_query}`，評估回應，並給出一個 10 分制的評分，只要輸出一個Float。",
            ),
            ("human", "{ai_response}"),
        ]
    )
    chain = prompt | llm
    overall_rating = chain.invoke(
        {
            "user_query": user_query,
            "ai_response": ai_response,
        }
    )
    try:
        overall_rating = float(overall_rating.content)
        rating_details = rating_details.content
    except:
        overall_rating = None
        rating_details = None
    return overall_rating, rating_details




# 當使用者提交新查詢時
if user_query := st.chat_input(placeholder="請輸入提問內容"):
    # 增加使用者的提問到聊天記錄
    st.session_state.messages.append(HumanMessage(user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.spinner("Searching knowledge base..."):
        time.sleep(1) 

        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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

        embedding_model = get_embedding_model("ollama")


        CONNECTION_STRING = "postgresql+psycopg2://biguser:npspo@10.96.196.63:5432/kmsdb"
        vector_store = PGVector(
            embeddings=embedding_model,
            collection_name='WIFI_802.11_Knowledge_Base',
            connection=CONNECTION_STRING,
        )


        id_key = 'wifi_doc_id' # 構建 Wifi知識庫的 ID

        # The retriever (empty to start)
        retriever = MultiVectorRetriever(
            vectorstore=vector_store,
            docstore=InMemoryStore(),
            id_key=id_key,
        )

        os.environ["COHERE_API_KEY"] = "fID3cnXZZpQzJfsErgBLuPu4mZfE1Tw9tcI4jSQP"
        compressor = CohereRerank(model="rerank-v3.5", top_n=10)


        llm = get_llm('llama3.1', 'ollama', 'http://10.96.196.63:11434/v1/', 0.6)

        test_retriever = retriever.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.6})
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=test_retriever
        )

        relevant_doc_lists = compression_retriever.invoke(user_query) # get_relevance_score deprecated


        context_list = []
        for i, relevant_doc in enumerate(relevant_doc_lists):
            print(f"Document {i}: {relevant_doc.metadata['remote_file']['file_name']}")
            print(f"Chunk {i}: {relevant_doc.page_content}")
            print(f"Relevance Score: {relevant_doc.metadata['relevance_score']}")
            context_list.append((relevant_doc.metadata['remote_file']['file_name'], relevant_doc.page_content, relevant_doc.metadata['relevance_score']))
            print('---------')

        reference_list = [thing[0] for thing in context_list]
        reference_list = "、".join(set(reference_list))
        print(reference_list)
        # formatted_context = "\n\n".join(context_chunks)

    # 檢查 OpenAI API 金鑰是否存在
    if not openai_api_key:
        st.info("請先輸入 OpenAI API Key")
        st.stop()

    with st.chat_message("AI"):

        ai_response = st.write_stream(new_get_response(user_query, compression_retriever))


        # 將 AI 回應後直接在同一個聊天訊息框內顯示參考資料
        with st.expander('📚 See Sources'):
            for i, (document_name, original_text, cosine_distance) in enumerate(context_list):
                st.markdown("**Source:**")
                file_path = f'./downloads/{document_name}'
                
                # 確保檔案存在 📄
                try:
                    with open(file_path, "rb") as file:
                        # 使用唯一的 key
                        unique_key = f"download_latest_{i}"
                        if st.download_button(label=f"📥 {document_name}", 
                                              data=file, 
                                              file_name=document_name, 
                                              key=unique_key):
                            pass  # 下載按鈕功能由 st.download_button 處理
                except FileNotFoundError:
                    st.warning(f"檔案 {document_name} 不存在")

                # Content 換行並加入 Tab 縮排
                st.markdown("**Content:**  \n" + f"&emsp;&emsp;{original_text}", unsafe_allow_html=True)
                st.write(
                    f'**Relavance Score：** {round(cosine_distance * 100, 2)}%'
                )
                st.divider()


        with st.spinner("Performing QA Assessment. Please wait..."):
            overall_rating, rating_details = evaluate_response(user_query, ai_response)
            st.write(f"🌟 **The Overall Rating is:**  {overall_rating} / 10")

        from datetime import date

        
        print(f'overall_rating: {overall_rating}')
        df = pd.DataFrame(
            {
                'username': ['DerekTY_Lin'],
                'session_id': [session_id],
                'thread_id': None,
                'model': [model],
                'user_query': [user_query],
                'ai_response': [ai_response],
                'prompt_template': [template],
                'reference_doc': [reference_list],
                'stable_version': [str(date.today())],
                'overall_rating': [overall_rating],
                'rating_details': [rating_details],
                'note': None,
                'insert_time': [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
            }
        )
        insert_and_eval_to_001GOD_BUDataLog(df)
    


    # 將 AI 回應和來源文檔保存到 session_state
    st.session_state.messages.append(AIMessage(ai_response))
    st.session_state.source_documents.append(context_list)
