from config import POSTGRES_URL
from sqlalchemy import create_engine, text


def get_pg_engine():
    """建立 PostgreSQL 連線並測試"""
    try:
        pg_engine = create_engine(POSTGRES_URL)
        with pg_engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))
            for row in result:
                print("✅ PostgreSQL 連線成功，版本資訊：", row[0])
        return pg_engine  # 只有當連線成功時才回傳引擎
    except Exception as e:
        print(f"❌ PostgreSQL 連線失敗: {e}")
        return None

def create_table():
    """建立 messages 表格"""
    engine = get_pg_engine()
    if engine is None:
        print("❌ 無法建立表格，資料庫連線失敗！")
        return

    with engine.connect() as connection:
        connection.execute(
            text("""
                CREATE TABLE IF NOT EXISTS wifi_knowledge_embedding_openai (
                    id SERIAL PRIMARY KEY,
                    document_name TEXT COLLATE pg_catalog."default" NOT NULL,
                    chunk_id INT NOT NULL,
                    original_text TEXT COLLATE pg_catalog."default",
                    cleaned_text TEXT COLLATE pg_catalog."default",
                    embedding VECTOR(1536), 
                    process_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT COLLATE pg_catalog."default",
                    metadata JSONB,
                    UNIQUE (document_name, chunk_id)
                );
            """)
        )
        connection.execute(
            text("""
                CREATE TABLE IF NOT EXISTS wifi_knowledge_embedding_bge (
                    id SERIAL PRIMARY KEY,
                    document_name TEXT COLLATE pg_catalog."default" NOT NULL,
                    chunk_id INT NOT NULL,
                    original_text TEXT COLLATE pg_catalog."default",
                    cleaned_text TEXT COLLATE pg_catalog."default",
                    embedding VECTOR(1024), 
                    process_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT COLLATE pg_catalog."default",
                    metadata JSONB,
                    UNIQUE (document_name, chunk_id)
                );
            """)
        )
        connection.commit()
        print("✅ 建立表格 wifi_knowledge_embedding_openai 成功")
        print("✅ 建立表格 wifi_knowledge_embedding_bge 成功")


# create_table()