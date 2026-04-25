from config import *

from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


# ========== ПРЕОБРАЗОВАНИЕ ДАННЫХ В ДОКУМЕНТЫ ==========
def create_documents_from_dicts(data: List[Dict]) -> List[Document]:
    """
    Преобразует список словарей в список Document объектов LangChain
    """
    documents = []

    for item in data:
        # Создаем текстовое представление словаря
        text = f"""
        Код родительской категории: {item.get('parent_code', '')}
        Название родительской категории: {item.get('parent_name', '')}
        Код элемента: {item.get('item_code', '')}
        Название элемента: {item.get('item_name', '')}
        """

        # Метаданные для поиска
        metadata = {
            "parent_code": item.get('parent_code', ''),
            "parent_name": item.get('parent_name', ''),
            "item_code": item.get('item_code', ''),
            "item_name": item.get('item_name', ''),
            "source": "dictionary_data"
        }

        doc = Document(page_content=text.strip(), metadata=metadata)
        documents.append(doc)

    return documents


# ========== РАЗБИЕНИЕ НА ЧАНКИ ==========
def split_documents(documents: List[Document]) -> List[Document]:
    """
    Разбивает документы на чанки
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Создано {len(chunks)} чанков из {len(documents)} документов")
    return chunks


# ========== RAG chain ==========


_PROMPT = ChatPromptTemplate.from_template("""\
    Ты - специалист по проверке кодов ОКВЭД.
    
    Контекст из документов:
    {context}
    
    Вопрос пользователя: {input}
    
    Дай развернутый ответ на основе контекста. Если информация не найдена, скажи об этом.
    Ответ:""")


def start_rag(db):
    """Создает RAG цепочку: retriever -> combine_docs -> retrieval."""

    retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_SEARCH_K})
    doc_chain = create_stuff_documents_chain(LLM, _PROMPT)

    print(f'RAG запущен')

    return create_retrieval_chain(retriever, doc_chain)
