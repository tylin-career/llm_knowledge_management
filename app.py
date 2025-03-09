from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from main import retrieve_similar_chunks
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
# from langchain_community.memory import ConversationBufferWindowMemory
from config import LLM_PROVIDER, OPENAI_API_KEY
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


st.set_page_config(page_title='Streamlit 知識管理對話系統')
st.title("💬 Chatbot")
st.caption("🚀 ASUS Knowledge Management Simulation Powered by NPSPO")

# 加入自訂 CSS，讓下拉選單展開時有動畫
st.markdown(
    """
    <style>
        /* 讓 selectbox 本身滑鼠懸停時有特效 */
        div[data-baseweb="select"] {
            transition: all 0.3s ease-in-out;
        }

        div[data-baseweb="select"]:hover {
            background-color: #f0f0f0 !important;
            border-radius: 8px;
        }

        /* 設定下拉選單本體 */
        div[role="listbox"] {
            animation: fadeIn 0.3s ease-in-out;
        }

        /* 定義動畫 */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.title("Navigation and Settings")
    model = st.selectbox(
        # 'Model', 'gpt-3.5-turbo'
        'Model', ['llama3.1', 'gpt-3.5-turbo']
    )
    openai_api_key = st.text_input(
        # 'OpenAI API Key', value = OPENAI_API_KEY, type = 'password'
        'OpenAI API Key', value = 'ollama', type = 'password'
    )
    openai_api_base = st.text_input(
        # 'OpenAI API Base', value = 'https://api.openai.com/v1/' # 'http://10.96.196.63:11434/v1/'
        'OpenAI API Base', value = 'http://10.96.196.63:11434/v1/'
    )
    temperature = st.slider(
        'Temperature', 0.0, 1.0, value = 0.6, step = 0.1
    )
    if st.sidebar.button('Clear Chat History'):
        st.session_state.clear()
        st.session_state["messages"] = []
        st.rerun()
    st.markdown('---')
    uploaded_file = st.file_uploader("📂 Upload Files", type=["doc", "docx", "txt", "md", "pdf"])


# 初始化聊天歷史
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "source_documents" not in st.session_state:
    st.session_state["source_documents"] = []


# 顯示歷史聊天記錄
for msg, src_docs in zip(st.session_state.messages, st.session_state.source_documents):
    if isinstance(msg, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("AI"):
            st.markdown(msg.content)
    with st.expander("Knowledge Base References"):
        for doc in src_docs:
            st.markdown(doc)
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


def get_response(user_query, formatted_context, chat_history):
    template = '''
        你是一位在 WiFi 6、WiFi 7 與 802.11 協議的專家，請根據參考資訊與對話紀錄回答問題：

        User question: {user_query}
        知識庫擷取的參考資訊：{formatted_context}
        Chat history: {chat_history}

        注意：
            1. 你需要依據提供的資訊與聊天紀錄回答，請勿編造內容。
            2. 若無足夠資訊，請回答「根據目前資訊無法回答」。
            3. 請以專業、精確的方式，以繁體中文為主回答問題。
    '''

    # https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm(model, openai_api_key, openai_api_base, temperature)
    chain = prompt | llm | StrOutputParser()
    return chain.stream(
        {
            'user_query': user_query,
            'formatted_context': formatted_context,
            'chat_history': chat_history
        }
    )


import time
# # 增加輸入框
if user_query := st.chat_input(placeholder="請輸入提問內容"):
    # 增加使用者的提問到聊天記錄
    st.session_state.messages.append(HumanMessage(user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)


    with st.spinner("Searching knowledge base..."):
        time.sleep(1.5)
        retrieved_data = retrieve_similar_chunks(user_query, "wifi_knowledge_embedding_bge", top_k=5)
        context_list = list(zip([context[1] for context in retrieved_data], [context[2] for context in retrieved_data]))
        # Get file_name and its remote path
        file_info_list = list(zip([document[0] for document in retrieved_data], [document[3] for document in retrieved_data]))
        context_chunks = [thing[0] for thing in context_list]
        formatted_context = "\n\n".join(context_chunks)
        # formatted_context = "some sample context" # self.generator.search_db(user_query)

    # 檢查 OpenAI API 金鑰是否存在
    if not openai_api_key:
        st.info("請先輸入 OpenAI API Key")
        st.stop()


    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query, formatted_context, st.session_state.messages))
    with st.expander('Knowledge Base References'):
        for i, (document_name, original_text, cosine_distance, file_path) in enumerate(retrieved_data):
            st.markdown("**Source:**")
            file_path = f'./downloads/{document_name}'
            
            # 確保檔案存在
            try:
                with open(file_path, "rb") as file:
                    # 按下按鈕時，更新 session_state
                    if st.download_button(label=f"📥 {document_name}", data=file, file_name=document_name, key=f"download_{i}"):
                        st.session_state["downloaded_files"][document_name] = True
            except FileNotFoundError:
                st.warning(f"檔案 {document_name} 不存在")

            # Content 換行並加入 Tab 縮排
            st.markdown("**Content:**  \n" + f"&emsp;&emsp;{original_text}", unsafe_allow_html=True)
            st.write(
                f'**Relavance Score：** {100 - round(cosine_distance * 100, 2)}%'
            )
            st.divider()

    st.session_state.messages.append(AIMessage(ai_response))
