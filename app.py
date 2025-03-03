from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler
# from langchain_community.memory import ConversationBufferWindowMemory
# from langchain_community.agents import ConversationalChatAgent, AgentExecutor
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from config import OPENAI_API_KEY
 
 
import streamlit as st
 
st.set_page_config(page_title='Streamlit 知識管理對話系統')
 
openai_api_base = st.sidebar.text_input(
    'OpenAI API Base', value = 'http://10.96.196.63:11434/v1/'
)
openai_api_key = st.sidebar.text_input(
    'OpenAI API Key', value = OPENAI_API_KEY, type = 'password'
)
 
model = st.sidebar.selectbox(
    'Model', ('Llama3.1', 'gpt-3.5-turbo')
)
 
temperature = st.sidebar.slider(
    'Temperature', 0.0, 2.0, value = 0.6, step = 0.1
)
 
 
 
message_history = StreamlitChatMessageHistory()
 
if not message_history.messages or st.sidebar.button('清空歷史紀錄'):
    message_history.clear()
    message_history.add_ai_message('歡迎來到知識問答對話系統')
 
    st.session_state.steps = {}
 
 
# 顯示歷史聊天記錄
for index, msg in enumerate(message_history.messages):
    with st.chat_message(msg.type):
        # 增加中間步驟到聊天視窗
        for step in st.session_state.steps.get(str(index), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        # 顯示對話內容
        st.write(msg.content)
 
# 增加輸入框
prompt = st.chat_input(placeholder="請輸入提問內容")
if prompt:
    # 檢查 OpenAI API 金鑰是否存在
    if not openai_api_key:
        st.info("請先輸入 OpenAI API Key")
        st.stop()
 
    # 增加使用者的提問到聊天記錄
    st.chat_message("human").write(prompt)
 
    # 建構 LLM Agent
    llm = ChatOpenAI(
        model=model,
        openai_api_key=openai_api_key,
        streaming=True,
        temperature=temperature,
        openai_api_base=openai_api_base,
    )
 
    tools = [DuckDuckGoSearchRun(name="Search")]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
 
    # 記憶體管理
    memory = ConversationBufferWindowMemory(
        chat_memory=message_history,
        return_messages=True,
        memory_key="chat_history",
        output_key="output",
        k=6,
    )
 
    # 執行 Agent
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
 
    # 增加 AI 回應
    with st.chat_message("ai"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = executor(prompt, callbacks=[st_cb])
        st.write(response["output"])
 
    # 儲存中間步驟
    step_index = str(len(message_history.messages) - 1)
    st.session_state.steps[step_index] = response["intermediate_steps"]