from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
# from langchain_community.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from main import get_prompt, retrieve_similar_chunks
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
from langchain.memory import ConversationBufferMemory
from config import LLM_PROVIDER, OPENAI_API_KEY
from dotenv import load_dotenv


load_dotenv()

def get_llm(model, openai_api_key, openai_api_base, temperature):
    if LLM_PROVIDER == "openai":
        print('使用 openai')
        return ChatOpenAI(
            model=model,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True,
        )
    elif LLM_PROVIDER == "ollama":
        print('使用 ollama')
        return ChatOpenAI(
            model=model,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            temperature=temperature,
            streaming=True,
        )

st.set_page_config(page_title='Streamlit 知識管理對話系統')
st.title("Chatbot")
 
model = st.sidebar.selectbox(
    'Model', ('llama3.1', 'gpt-3.5-turbo')
)
openai_api_key = st.sidebar.text_input(
    'OpenAI API Key', value = 'ollama'
    # 'OpenAI API Key', value = OPENAI_API_KEY, type = 'password'
)
openai_api_base = st.sidebar.text_input(
    'OpenAI API Base', value = 'http://10.96.196.63:11434/v1/'
)
temperature = st.sidebar.slider(
    'Temperature', 0.0, 1.0, value = 0.6, step = 0.1
)

streamlit_key = 'Derek'
# 使用 st.session_state 儲存聊天歷史，避免重新渲染時重建物件
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = StreamlitChatMessageHistory(key=streamlit_key)
history = st.session_state["chat_history"]


llm = get_llm(model, openai_api_key, openai_api_base, temperature)
prompt = get_prompt()
memory = ConversationBufferMemory(chat_memory=history, memory_key="conversation", return_messages=True)
chain = prompt | llm



if not history.messages or len(history.messages) == 0 or st.sidebar.button('清空歷史紀錄'):
    history.clear()
    history.add_ai_message('歡迎來到知識問答對話系統')





# 顯示歷史聊天記錄
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

def get_response(user_query, formatted_context):
    return chain.stream(
        {
            'user_query': user_query,
            'context': formatted_context,
            "conversation": memory.load_memory_variables({})["conversation"]
        }
    )

 
# 增加輸入框
if user_query := st.chat_input(placeholder="請輸入提問內容"):
    # 檢查 OpenAI API 金鑰是否存在
    if not openai_api_key:
        st.info("請先輸入 OpenAI API Key")
        st.stop()


    memory.chat_memory.add_user_message(user_query)
    # 增加使用者的提問到聊天記錄
    st.chat_message("human").write(user_query)


    retrieved_data = retrieve_similar_chunks(user_query, "wifi_knowledge_embedding_bge", top_k=5)
    context_list = list(zip([context[1] for context in retrieved_data], [context[2] for context in retrieved_data]))
    # Get file_name and its remote path
    file_info_list = list(zip([document[0] for document in retrieved_data], [document[3] for document in retrieved_data]))
    context_chunks = [thing[0] for thing in context_list]
    formatted_context = "\n\n".join(context_chunks)



    ai_reponses = get_response(user_query, formatted_context)

    st.chat_message("ai").write(ai_reponses)
    # 將完整的 AI 回應加入記憶體中，確保下次渲染時依然存在
    memory.chat_memory.add_ai_message(ai_reponses)

        

    # # 增加 AI 回應
    # with st.chat_message("ai"):
    #     st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    #     response = llm.invoke({"messages": prompt}, config={"configurable": {"thread_id": 42}})

    #     st.write(response["output"])
 