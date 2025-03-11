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


from dev_config import Configuration


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





def get_llm(config:Configuration):
    LLM_PROVIDER = config.LLM_PROVIDER
    OPENAI_API_KEY = config.OPENAI_API_KEY


    print(f'使用 {LLM_PROVIDER} 模型')
    if LLM_PROVIDER == "gpt-3.5-turbo":
        return ChatOpenAI(
            model=LLM_PROVIDER,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True,
        )
    elif LLM_PROVIDER == "llama3.1":
        return ChatOpenAI(
            model=LLM_PROVIDER,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=openai_api_base,
            temperature=temperature,
            streaming=True,
        )


def main():
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


    config = Configuration()

    # init llm
    llm = get_llm(config)

if __name__ == '__main__':
    main()
