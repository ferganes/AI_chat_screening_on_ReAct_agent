from langchain_core.tools import tool

# Глобальная переменная для vectorstore, устанавливается при запуске приложения
_vectorstore = None


def set_vectorstore(vs):
    """Устанавливает глобальный vectorstore"""
    global _vectorstore
    _vectorstore = vs


@tool
def approve_user() -> str:
    """Подключение пользователя к сервису. Используется ТОЛЬКО когда все 3 условия выполнены."""
    return "Соответствие требованиям подключения. Пройдите по ссылке для завершения подключения к сервису."


@tool
def reject_user() -> str:
    """Отказ в подключение пользователя к сервису. Используется при невыполнении условий"""
    return "К сожалению, вы несоответствуете требованиям подключения."


@tool
def search_description_in_okved(query: str) -> str:
    """Проверяет наличие переданного пользователем описания бизнеса в описании кодов ОКВЭД"""

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
        return f"РЕЗУЛЬТАТ: Ошибка поиска описания. {str(e)}. Попробуйте переформулировать описание бизнеса."


def get_tools():
    return [approve_user, reject_user, search_description_in_okved]
