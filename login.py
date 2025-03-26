import streamlit as st

def main():
    st.session_state.clear()
    st.title("使用者登入")
    username = st.text_input("請輸入帳號")
    
    if st.button("登入"):
        if username:
            st.session_state["username"] = username
            # 使用 meta refresh 導向位於 pages 資料夾的 app.py 頁面
            st.markdown(
                f"""
                <meta http-equiv="refresh" content="0;url=./pages/app.py">
                如果沒有自動跳轉，請點<a href="./pages/app.py">這裡</a>.
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("請輸入有效的帳號！")

if __name__ == '__main__':
    main()