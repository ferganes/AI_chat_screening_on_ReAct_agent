import time
import json
import shutil
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PERSIST_DIR, COLLECTION, EMBEDDINGS, CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS

_db_instance = None


def split_documents(documents):
    """Разбивает документы на чанки"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def create_database(force_recreate: bool = False):
    """Создает базу данных Chroma из okved.json с контролем скорости"""
    global _db_instance

    try:
        if force_recreate:
            # Удаляем существующую базу
            if os.path.exists(PERSIST_DIR):
                shutil.rmtree(PERSIST_DIR)
            _db_instance = None

        # Проверяем, существует ли уже БД
        if not force_recreate and os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            # БД уже существует, просто подключаемся
            return connect_database()

        # Загружаем данные из JSON
        print(f"Загрузка данных из okved.json...")
        documents = load_okved_data()
        print(f"Загружено {len(documents)} документов")

        # Разбиваем на чанки
        chunks = split_documents(documents)
        print(f"Создано {len(chunks)} чанков")

        # Создаем базу данных пакетами с задержкой
        batch_size = 10  # YandexGPT лимит - 10 запросов в секунду
        print(f"Создание базы данных пакетами по {batch_size} документов...")

        _db_instance = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=EMBEDDINGS,
            collection_name=COLLECTION
        )

        # Добавляем документы пакетами
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"Обработка пакета {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} "
                  f"(документы {i + 1}-{min(i + batch_size, len(chunks))})")

            _db_instance.add_documents(batch)

            # Ждем 1 секунду между пакетами (10 запросов в секунду)
            if i + batch_size < len(chunks):
                time.sleep(1)

        print(f"База данных успешно создана в {PERSIST_DIR}")

        return _db_instance

    except Exception as e:
        print(f"Ошибка при создании базы данных: {e}")
        raise


def connect_database():
    """Подключается к существующей БД и возвращает vectorstore"""
    global _db_instance

    if _db_instance is None:
        _db_instance = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=EMBEDDINGS,
            collection_name=COLLECTION
        )
        print(f"Подключено к существующей базе данных в {PERSIST_DIR}")

    return _db_instance


def get_vectorstore():
    """Возвращает текущий vectorstore"""
    return _db_instance


def initialize_database(force_recreate: bool = False):
    """
    Инициализирует базу данных: создает новую или подключается к существующей
    force_recreate=True - удаляет существующую БД и создает заново
    """
    return create_database(force_recreate) if force_recreate else connect_database()