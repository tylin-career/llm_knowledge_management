import streamlit as st
from main import qa_query

# 使用 st.session_state 儲存對話歷史
if "conversation" not in st.session_state:
    st.session_state.conversation = []

st.title("WiFi知識問答系統")

user_query = st.text_input("請輸入你的問題：")

if st.button("送出查詢") and user_query:
    # 將使用者提問加入對話歷史
    st.session_state.conversation.append({"role": "user", "content": user_query})
    
    # 呼叫 main.py 中的 qa_query 函式取得 AI 回答的串流產生器及參考資訊
    response_generator, context_info = qa_query(user_query, st.session_state.conversation)
    
    st.subheader("檢索到的參考資訊")
    st.text(context_info)
    
    st.subheader("AI 回答")
    placeholder = st.empty()
    full_response = ""
    
    # 串流更新 AI 回答
    for ai_reply in response_generator:
        full_response += ai_reply.content
        placeholder.text(full_response)
    
    # 將 AI 回答加入對話歷史
    st.session_state.conversation.append({"role": "ai", "content": full_response})