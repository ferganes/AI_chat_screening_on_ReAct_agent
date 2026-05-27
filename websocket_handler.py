from langchain_core.messages import HumanMessage, AIMessage
from config import MAX_CHAT_HISTORY


async def agent_process_message(agent, user_message: str, chat_history: list):
    """Обработка сообщения пользователя через LangGraph ReAct."""

    response = await agent.ainvoke({
        "messages": [*chat_history, HumanMessage(content=user_message)]
    })

    last_message = response["messages"][-1]
    answer = getattr(last_message, "content", str(last_message))

    chat_history.extend([
        HumanMessage(content=user_message),
        AIMessage(content=answer),
    ])

    # Ограничиваем размер истории (оставляем последние MAX_CHAT_HISTORY пар)
    if len(chat_history) > MAX_CHAT_HISTORY * 2:
        chat_history[:] = chat_history[-MAX_CHAT_HISTORY * 2:]

    return answer, chat_history
