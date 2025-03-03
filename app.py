from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
# from langchain_community.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage
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
 
    from langchain_core.messages import trim_messages
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import START, MessagesState, StateGraph

    # Define trimmer
    # count each message as 1 "token" (token_counter=len) and keep only the last two messages
    trimmer = trim_messages(strategy="last", max_tokens=2, token_counter=len)

    workflow = StateGraph(state_schema=MessagesState)


    # Define the function that calls the model
    def call_model(state: MessagesState):
        trimmed_messages = trimmer.invoke(state["messages"])
        system_prompt = (
            "You are a helpful assistant. "
            "Answer all questions to the best of your ability."
        )
        messages = [SystemMessage(content=system_prompt)] + trimmed_messages
        response = model.invoke(messages)
        return {"messages": response}


    # Define the node and edge
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    # Add simple in-memory checkpointer
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
 

 
    # 增加 AI 回應
    with st.chat_message("ai"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = app.invoke({"messages": prompt}, config={"configurable": {"thread_id": 42}})

        st.write(response["output"])
 
    # 儲存中間步驟
    step_index = str(len(message_history.messages) - 1)
    st.session_state.steps[step_index] = response["intermediate_steps"]