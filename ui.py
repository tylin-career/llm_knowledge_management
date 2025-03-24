from config import EmbeddingConfiguration, LLMConfiguration
import streamlit as st
import os
from src.generator import Generator
from src.ragger import RAGGER
from src.timer import timing_decorator



class UI:
    def __init__(self):
        pass


    def show_main_page(self):
        st.title("知識查詢 🌐️")
    def show_knowledge_base_page(self):
        st.title("檔案管理與上傳")

    def main(self):
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