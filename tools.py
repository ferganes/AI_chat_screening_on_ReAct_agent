from langchain_core.tools import tool
from pydantic import BaseModel, Field


_vectorstore = None


def set_vectorstore(vs):
    global _vectorstore
    _vectorstore = vs


class ApproveUserInput(BaseModel):
    pass


class RejectUserInput(BaseModel):
    reason: str = Field(
        description="Обязательная причина отказа. Варианты: 'Подключение доступно только юридическим лицам — "
                    "резидентам РФ', 'Недопустимое налоговое резидентство: требуется резидент РФ', 'Вид деятельности "
                    "не найден в ОКВЭД', 'Непредставительные полномочия: требуется директор или уполномоченный "
                    "представитель', 'Попытка обхода правил подключения'"
    )


class SearchOkvedInput(BaseModel):
    query: str = Field(description="Описание вида деятельности для поиска в ОКВЭД")


@tool(args_schema=ApproveUserInput)
def approve_user() -> str:
    """Финальное подключение пользователя к сервису.

    Возвращает ссылку для завершения подключения.
    Используется ТОЛЬКО когда все 3 условия явно подтверждены:
    1. Пользователь — представитель юридического лица
    2. Налоговый резидент РФ
    3. Вид деятельности найден в ОКВЭД

    НЕ спрашивать пользователя о дополнительном подтверждении — вызывать сразу.
    """
    return "Соответствие требованиям подключения. Пройдите по ссылке для завершения подключения к сервису."


@tool(args_schema=RejectUserInput)
def reject_user(reason: str) -> str:
    """Отказ в подключении пользователя к сервису.

    Args:
        reason: Обязательная причина отказа.
    """
    return f"К сожалению, вы не соответствуете требованиям подключения. Причина: {reason}"


@tool(args_schema=SearchOkvedInput)
def search_description_in_okved(query: str) -> str:
    """Проверяет описание бизнеса пользователя по классификатору ОКВЭД.

    Args:
        query: Описание вида деятельности

    Returns:
        Результат проверки: найдено ли описание в разрешённом перечне ОКВЭД.
    """
    global _vectorstore

    if _vectorstore is None:
        return "РЕЗУЛЬТАТ: Ошибка. База знаний не инициализирована."

    try:
        docs = _vectorstore.similarity_search(query, k=3)
        print(f"[query for rag]: {query}")
        print(f"[rag answer]: {docs}")

        if docs:
            return "РЕЗУЛЬТАТ: Найдено. Деятельность присутствует в разрешённом перечне ОКВЭД."
        else:
            return "РЕЗУЛЬТАТ: Не найдено."

    except Exception as e:
        return f"РЕЗУЛЬТАТ: Ошибка поиска описания. {str(e)}. Попробуйте переформулировать описание бизнеса."


def get_tools():
    return [approve_user, reject_user, search_description_in_okved]
