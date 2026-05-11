from langgraph.prebuilt import create_react_agent

from config import LLM
from prompt import SYSTEM_PROMPT
from tools import get_tools


def create_agent(llm):
    """Создает и возвращает агента"""
    return create_react_agent(
        model=LLM,
        tools=get_tools(),
        prompt=SYSTEM_PROMPT
    )
