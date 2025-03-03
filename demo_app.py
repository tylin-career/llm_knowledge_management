import streamlit as st
from langchain_openai import ChatOpenAI
from config import LLM_PROVIDER, OPENAI_API_KEY
from embedding import get_embedding_model
from postgresql import get_pg_engine
from sqlalchemy import text

# 初始化 LLM
def get_llm():
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
            api_key=OPENAI_API_KEY,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True,
        )
    elif LLM_PROVIDER == "ollama":
        return ChatOpenAI(
            model="llama3.1",
            openai_api_key="ollama",
            openai_api_base="http://10.96.196.63:11434/v1/",
            streaming=True,
        )

# 初始化嵌入模型
embedding_model = get_embedding_model()
# 初始化 PostgreSQL
pg_engine = get_pg_engine()


def search_relevant_docs(query_text, top_k=3):
    """ 在資料庫中檢索與 query_text 最相似的文件 """
    query_embedding = embedding_model.embed_query(query_text)

    table_name = "wifi_knowledge_embedding_bge"  # 若使用 OpenAI，改為 `wifi_knowledge_embedding_openai`
    
    sql = f"""
    SELECT 
        document_name, 
        original_text, 
        file_path, 
        embedding <=> :query_embedding AS similarity
    FROM {table_name}
    ORDER BY similarity ASC
    LIMIT :top_k;
    """

    with pg_engine.connect() as connection:
        result = connection.execute(text(sql), {"query_embedding": query_embedding, "top_k": top_k})
        docs = result.fetchall()

    return docs

def format_context(docs):
    """ 格式化檢索到的文檔內容 """
    context = "\n\n".join([f"📄 **檔案**: {doc[0]}\n🔍 **內容**: {doc[1][:200]}...\n📂 **來源**: {doc[2]}" for doc in docs])
    return f"### 🔎 相關文件內容:\n\n{context}"

def answer_question(query):
    """ 透過 RAG 檢索資料並讓 LLM 回答 """
    docs = search_relevant_docs(query)
    context = format_context(docs)

    messages = [
        ("system", "你是一個知識管理助理，請根據提供的內容回答使用者的問題。"),
        ("system", context),
        ("human", query),
    ]

    llm = get_llm()
    response = ""
    for chunk in llm.stream(messages):
        response += chunk.content
        yield chunk.content  # 逐步回傳給 Streamlit 前端










# ---------- Streamlit 介面 ----------
st.set_page_config(page_title="知識管理系統 RAG", page_icon="📚", layout="wide")

st.title("📚 知識管理系統 (RAG)")
st.write("請輸入您的問題，系統將從知識庫中檢索相關內容並回答您的問題。")

# 使用者輸入
user_input = st.text_area("💬 輸入您的問題:", height=150)

# 送出查詢
if st.button("🔍 查詢"):
    if user_input.strip():
        with st.spinner("📖 正在檢索知識庫並生成回答..."):
            # 顯示檢索結果
            docs = search_relevant_docs(user_input)
            if docs:
                st.markdown(format_context(docs))
            else:
                st.warning("⚠️ 沒有找到相關的知識。")

            # 生成答案
            st.markdown("### 🤖 AI 回答:")
            response_area = st.empty()  # 建立一個區域來顯示回應
            response_text = ""

            for chunk in answer_question(user_input):
                response_text += chunk
                response_area.markdown(response_text)
    else:
        st.warning("⚠️ 請輸入問題後再查詢。")




'''
import chromadb
from sentence_transformers import SentenceTransformer
import openai  # 如果你要使用 OpenAI API
import ollama  # 如果你要用本機 Ollama（支援 LLaMA 3.1）

# 1️⃣ 初始化 ChromaDB 客戶端
chroma_client = chromadb.PersistentClient(path="./chroma_langchain_db")
collection = chroma_client.get_or_create_collection(name="documents")

# 2️⃣ 初始化向量模型（使用 `sentence-transformers`）
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 輕量級但準確

# 3️⃣ 定義 RAG 查詢函式
def query_rag(user_input):
    # 👉 取得使用者輸入的 embedding
    query_embedding = embedding_model.encode(user_input).tolist()

    # 👉 用向量搜尋 ChromaDB（取最相關的 3 筆資料）
    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    # 👉 組合檢索到的內容
    retrieved_texts = [doc[0] if isinstance(doc, list) else doc for doc in search_results["documents"]]
    context = "\n".join(retrieved_texts)

    # 🔹 使用 OpenAI API（如果你有 API Key）
    response = openai.ChatCompletion.create(
        model="gpt-4",  # 或 "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "你是一個知識庫助手，請使用提供的文件來回答問題。"},
            {"role": "user", "content": f"文件內容：\n{context}\n\n使用者問題：{user_input}"}
        ]
    )

    return response["choices"][0]["message"]["content"]

# 4️⃣ 測試 RAG 查詢
user_question = input("請輸入你的問題：")
answer = query_rag(user_question)
print("\n💡 AI 回答：\n", answer)


'''