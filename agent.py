from langgraph.prebuilt import create_react_agent
from prompt import SYSTEM_PROMPT
from tools import get_tools


def create_agent(llm):
    """Создает и возвращает агента"""
    return create_react_agent(
        model=llm,
        tools=get_tools(),
        prompt=SYSTEM_PROMPT
    )
