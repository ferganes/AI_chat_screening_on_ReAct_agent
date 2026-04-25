import os
import shutil
from langchain_chroma import Chroma

from config import PERSIST_DIR, COLLECTION, EMBEDDINGS


def check_database_exists():
    """
    Проверяет существование директории и файла базы данных.
    Возвращает True, если всё ок. False, если всё не ок.
    """
    if not os.path.exists(PERSIST_DIR):
        return False

    db_file = os.path.join(PERSIST_DIR, "chroma.sqlite3")
    return os.path.exists(db_file)


def create_database():
    """
    Создаёт директорию для новой БД.
    Возвращает True, если всё ок. False, если всё не ок.
    """
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)
        print(f"--> Создана директория базы данных: {PERSIST_DIR}")
        return True
    return False



def update_database(db, docs):
    """
    Добавляет документы в базу данных

    Args:
        db: ChromaDB
        docs: список Document
    """

    db.add_documents(docs)


def drop_database():
    shutil.rmtree(PERSIST_DIR)
    print(f"--> БД в директории {PERSIST_DIR} удалена...")
