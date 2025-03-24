from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
# from langchain_community.memory import ConversationBufferWindowMemory
from config import OPENAI_API_KEY
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config import LLMConfiguration


st.set_page_config(page_title='Streamlit 知識管理對話系統', page_icon='📊', layout='wide', initial_sidebar_state='expanded')
st.title("💬 Chatbot")
st.caption("🚀 ASUS Knowledge Management Simulation Powered by NPSPO")


# 初始化聊天歷史
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "source_documents" not in st.session_state:
    st.session_state["source_documents"] = []


with st.sidebar:
    with st.form(key='my_form'):
        st.title("Navigation and Settings")
        st.caption("🔧 這是側邊欄的內容")
        model = st.selectbox(
            # 'Model', 'gpt-3.5-turbo'
            'Model', ['llama3.1', 'gpt-3.5-turbo']
        )
        if model == "gpt-3.5-turbo":
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



def get_llm(model, openai_api_key, openai_api_base, temperature):
    print(f'使用 {model}')
    if model == "gpt-3.5-turbo":
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


def get_response(user_query, formatted_context):
    
    template = """你是一位在 WiFi 6、WiFi 7 與 802.11 協議的專家，請根據參考資訊與對話紀錄回答問題：
    User question: {user_query}
    知識庫擷取的參考資訊：{formatted_context}
    若無足夠資訊，請回答「根據目前資訊無法回答」。
    若有足夠參考資訊，請用繁體中文回答問題，並調整適當輸出格式，
    請先針對回答做概述，然後直接回應，然後針對回答做出額外闡釋，然後以條列式總結回應
    """
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

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


import time
# 當使用者提交新查詢時
if user_query := st.chat_input(placeholder="請輸入提問內容"):
    # 增加使用者的提問到聊天記錄
    st.session_state.messages.append(HumanMessage(user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.spinner("Searching knowledge base..."):
        # time.sleep(2.5)
        # retrieved_data = retrieve_similar_chunks(user_query, "wifi_knowledge_embedding_bge", top_k=5)
        # context_list = list(zip([context[1] for context in retrieved_data], [context[2] for context in retrieved_data]))
        # # Get file_name and its remote path
        # file_info_list = list(zip([document[0] for document in retrieved_data], [document[3] for document in retrieved_data]))
        # context_chunks = [thing[0] for thing in context_list]
        time.sleep(1)



        from langchain.retrievers.multi_vector import MultiVectorRetriever
        from langchain_core.documents import Document
        import os
        from langchain_ollama import OllamaEmbeddings
        from langchain_openai import OpenAIEmbeddings

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
        from langchain_postgres.vectorstores import PGVector


        from langchain_core.stores import InMemoryStore

        # CONNECTION_STRING = "postgresql+psycopg2://biguser:npspo@10.96.196.64:32/kmsdb"
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
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
        from langchain_cohere.rerank import CohereRerank
        os.environ["COHERE_API_KEY"] = "fID3cnXZZpQzJfsErgBLuPu4mZfE1Tw9tcI4jSQP"
        compressor = CohereRerank(model="rerank-v3.5", top_n=10)

        from langchain_openai import ChatOpenAI
        def get_llm(model, openai_api_key, openai_api_base, temperature):
            print(f'使用 {model}')
            if model == "gpt-3.5-turbo":
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

        context_chunks = [thing[1] for thing in context_list]



        formatted_context = "\n\n".join(context_chunks)

    # 檢查 OpenAI API 金鑰是否存在
    if not openai_api_key:
        st.info("請先輸入 OpenAI API Key")
        st.stop()

    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query, formatted_context))
        
        # 將 AI 回應後直接在同一個聊天訊息框內顯示參考資料
        with st.expander('See Sources'):
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

    # 將 AI 回應和來源文檔保存到 session_state
    st.session_state.messages.append(AIMessage(ai_response))
    st.session_state.source_documents.append(context_list)
