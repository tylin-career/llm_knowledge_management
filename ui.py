from config import EmbeddingConfiguration, LLMConfiguration
import streamlit as st
import os
from src.generator import Generator
from src.ragger import RAGGER
from src.timer import timing_decorator

USER_CREDENTIALS = {
    "username": "admin",
    "password": "password"
}

class UI:
    def __init__(self):
        if "logged_in_ok" not in st.session_state:
            st.session_state.logged_in_ok = False


    def login_page(self):
        st.title("🔐 使用者登入")

        username = st.text_input("帳號")
        password = st.text_input("密碼", type="password")
        login_button = st.button("登入")

        if login_button:
            if username in ('admin') and password == USER_CREDENTIALS[password]:
                st.session_state.logged_in_ok = True
                st.session_state.username = username
                st.success(f"{username} 登入成功！")
                st.rerun()
            else:
                st.error("帳號或密碼錯誤")

    def show_main_page(self):
        st.title("知識查詢 🌐️")
        st.write(f"歡迎，{st.session_state.username}！您已成功登入。")
    def show_knowledge_base_page(self):
        st.title("檔案管理與上傳")

    def main(self):
        if not st.session_state.logged_in_ok:
            self.login_page()
        else:
            st.sidebar.title("Navigation")
            page = st.sidebar.selectbox(
                "前往", ("知識查詢", "檔案管理與上傳")
            )
            if page == "知識查詢":
                self.show_main_page()
            elif page == "檔案管理與上傳":
                self.show_knowledge_base_page()


if __name__ == '__main__':
    UI().main()