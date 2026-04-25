from langchain_core.tools import tool

# Глобальная переменная для vectorstore (устанавливается при старте)
_vectorstore = None


def set_vectorstore(vs):
    """Устанавливает глобальный vectorstore"""
    global _vectorstore
    _vectorstore = vs


@tool
def connect_to_service() -> str:
    """Подключает пользователя к сервису. Используется ТОЛЬКО когда все 3 условия явно подтверждены."""

    return ("Ваш бизнес соответствует требованиям. Подключаем вас к сервису. Ваша ссылка для подключения: "
            "https://best-service-ever.com")


@tool
def knowledge_base_search(query: str) -> str:
    """Ищет релевантную информацию по ОКВЭД и описанию ОКВЭД во внутренней базе знаний RAG."""

    global _vectorstore

    if _vectorstore is None:
        return "РЕЗУЛЬТАТ: Ошибка. База знаний не инициализирована."

    try:
        docs = _vectorstore.similarity_search(query, k=3)

        if docs:
            print(f"[query for rag]: {query}")
            print(f"[rag answer]: {docs}")
            return "РЕЗУЛЬТАТ: Найдено. Деятельность присутствует в разрешенном перечне ОКВЭД."
        else:
            print(f"[query for rag]: {query}")
            print(f"[rag answer]: {docs}")
            return "РЕЗУЛЬТАТ: Не найдено."

    except Exception as e:
        return f"РЕЗУЛЬТАТ: Ошибка поиска. {str(e)}. Попробуйте переформулировать описание бизнеса."


def get_tools():
    return [connect_to_service, knowledge_base_search]