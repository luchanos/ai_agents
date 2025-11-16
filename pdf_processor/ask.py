import sys
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Константы ---
CHROMA_PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"  # Используем дешевый и быстрый gpt-3.5

# --- Шаблон Промпта ---
# Это сердце нашего RAG.
# Мы говорим LLM, как себя вести.
PROMPT_TEMPLATE = """
Ты — вежливый и точный ИИ-ассистент, специализирующийся на ответах 
по технической документации.

Твоя задача — отвечать на вопрос пользователя, 
основываясь **исключительно** на предоставленном ниже контексте.

**Контекст:**
{context}

**Вопрос:**
{question}

**Инструкции:**
1.  Внимательно прочти контекст.
2.  Найди в нем ответ на вопрос.
3.  Если ответ есть в контексте, сформулируй его своими словами. 
    Не копируй текст дословно.
4.  **Если в контексте нет ответа на вопрос**, вежливо скажи: 
    "К сожалению, я не нашел точной информации по этому вопросу 
    в предоставленной документации."
5.  Не используй свои общие знания. Не выдумывай.
"""


def main():
    print("Загрузка переменных окружения...")
    load_dotenv()

    # --- 1. Инициализация компонентов ---
    print("Загрузка LLM, векторизатора и базы данных...")

    # LLM ("Мозг")
    llm = ChatOpenAI(model_name=LLM_MODEL)

    # Векторизатор (тот же, что при индексации)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Подключение к существующей базе
    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )

    # Ретривер (компонент для поиска)
    # k=3 означает, что мы будем доставать 3 самых релевантных чанка
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Промпт
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # --- 2. Сборка Цепочки (Chain) ---
    # Мы используем LangChain Expression Language (LCEL)
    # Это похоже на "пайп" | в Linux.

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print("Агент готов к работе. Введите 'выход' или 'exit' для завершения.")

    # --- 3. Цикл вопросов-ответов ---
    while True:
        try:
            query = input("\nВаш вопрос: ")
            if query.lower() in ["выход", "exit"]:
                print("До свидания!")
                sys.exit()

            if not query.strip():
                continue

            print("Думаю...")

            # Запускаем цепочку
            answer = rag_chain.invoke(query)

            print("\nОтвет:")
            print(answer)

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nДо свидания!")
            break


if __name__ == "__main__":
    main()