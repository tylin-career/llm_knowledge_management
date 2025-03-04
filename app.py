from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from main import get_prompt, retrieve_similar_chunks
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
# from langchain_community.memory import ConversationBufferWindowMemory
from config import LLM_PROVIDER, OPENAI_API_KEY

st.set_page_config(page_title='Streamlit 知識管理對話系統')
st.title("💬 Chatbot")
st.caption("🚀 ASUS Knowledge Management Simularted by NPSPO")

with st.sidebar:
    model = st.selectbox(
        'Model', ('llama3.1', 'gpt-3.5-turbo')
    )
    openai_api_key = st.text_input(
        'OpenAI API Key', value = 'ollama', type = 'password'
    )
    openai_api_base = st.text_input(
        'OpenAI API Base', value = 'http://10.96.196.63:11434/v1/'
    )
    temperature = st.slider(
        'Temperature', 0.0, 1.0, value = 0.6, step = 0.1
    )


# 初始化聊天歷史
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "歡迎來到知識問答系統"}]

# 顯示歷史聊天記錄
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# streamlit_key = 'Derek'
# # 使用 st.session_state 儲存聊天歷史，避免重新渲染時重建物件
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = StreamlitChatMessageHistory(key=streamlit_key)
# history = st.session_state["chat_history"]


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

# 清除 session 並重設 messages
if st.sidebar.button('清空歷史紀錄'):
    st.session_state.clear()  # 清除所有 session state
    st.session_state["messages"] = [{"role": "assistant", "content": "歡迎來到知識問答系統"}]



# if not st.session_state.messages or len(st.session_state.messages) == 0 or st.sidebar.button('清空歷史紀錄'):
#     st.session_state.clear()
#     st.session_state.add_ai_message('歡迎來到知識問答對話系統')



# def get_response(user_query, formatted_context):
#     return chain.stream(
#         {
#             'user_query': user_query,
#             'context': formatted_context,
#             "conversation": memory.load_memory_variables({})["conversation"]
#         }
#     )

def get_response(user_query, formatted_context):
    return 'AI 回答'

# # 增加輸入框
if user_query := st.chat_input(placeholder="請輸入提問內容"):
    # 檢查 OpenAI API 金鑰是否存在
    if not openai_api_key:
        st.info("請先輸入 OpenAI API Key")
        st.stop()
    llm = get_llm(model, openai_api_key, openai_api_base, temperature)
    prompt = get_prompt()
    # memory = ConversationBufferMemory(chat_memory=history, memory_key="conversation", return_messages=True)
    chain = prompt | llm


    # 增加使用者的提問到聊天記錄
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)


    formatted_context = "context"
    ai_response = get_response(user_query, formatted_context)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.chat_message("assistant").write(ai_response)


#     retrieved_data = retrieve_similar_chunks(user_query, "wifi_knowledge_embedding_bge", top_k=5)
#     context_list = list(zip([context[1] for context in retrieved_data], [context[2] for context in retrieved_data]))
#     # Get file_name and its remote path
#     file_info_list = list(zip([document[0] for document in retrieved_data], [document[3] for document in retrieved_data]))
#     context_chunks = [thing[0] for thing in context_list]
#     formatted_context = "\n\n".join(context_chunks)


#     ai_reponses = get_response(user_query, formatted_context)
#     st.chat_message("ai").write(ai_reponses)
