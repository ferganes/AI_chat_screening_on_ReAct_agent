from config import *

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# RAG цепочка

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
