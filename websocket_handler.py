from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from prompt import SYSTEM_PROMPT
from config import MAX_CHAT_HISTORY


async def agent_process_message(agent, user_message: str, chat_history: list):
    """Обработка сообщения пользователя"""
    print(f"[USER MESSAGE] Получено сообщение: {user_message}")

    # Формируем контекст для llm
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=user_message))

    # Вызываем агента
    response = await agent.ainvoke({"messages": messages})
    answer = response["messages"][-1].content

    print(f"[LLM ANSWER] Ответ: {answer}")

    # Обновляем историю
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=answer))

    # Ограничиваем размер истории
    if len(chat_history) > MAX_CHAT_HISTORY * 2:
        chat_history[:] = chat_history[-MAX_CHAT_HISTORY * 2:]

    return answer, chat_history
