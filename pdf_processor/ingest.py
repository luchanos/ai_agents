import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Константы ---
# Путь к папке с PDF
DOCS_DIR = "docs"
# Путь к локальной базе ChromaDB
CHROMA_PERSIST_DIR = "chroma_db"
# Модель для векторизации (быстрая, локальная)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Параметры "нарезки" текста
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def main():
    print("Загрузка переменных окружения...")
    load_dotenv()

    # --- 1. Загрузка Документов ---
    print(f"Загрузка документов из {DOCS_DIR}...")
    documents = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DOCS_DIR, filename)
            loader = PyMuPDFLoader(filepath)
            documents.extend(loader.load())

    if not documents:
        print("Документы не найдены. Убедитесь, что в папке 'docs' есть PDF-файлы.")
        return

    print(f"Загружено {len(documents)} страниц.")

    # --- 2. Нарезка (Chunking) ---
    print("Нарезка документов на чанки...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Создано {len(chunks)} чанков.")

    # --- 3. Векторизация (Embedding) ---
    print("Инициализация модели векторизации...")
    # Используем локальную модель.
    # При первом запуске она скачается (около 90 МБ).
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # --- 4. Сохранение в Векторную БД ---
    print("Сохранение чанков и векторов в ChromaDB...")

    # Создаем (или перезаписываем) базу данных в локальной папке
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

    print(f"База данных успешно создана и сохранена в '{CHROMA_PERSIST_DIR}'.")
    print("Индексация завершена!")


if __name__ == "__main__":
    main()