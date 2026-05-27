from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage

from config import LLM
from prompt import SYSTEM_PROMPT
from tools import get_tools


def create_agent():
    tools = get_tools()
    tool_node = ToolNode(tools)
    llm_with_tools = LLM.bind_tools(tools)

    # Формируем контекст из системного промпта и предыдущих сообщений в чате
    async def call_llm(state: MessagesState):
        msgs = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        response = await llm_with_tools.ainvoke(msgs)
        return {"messages": [response]}

    def route(state: MessagesState):
        return "tools" if state["messages"][-1].tool_calls else END

    graph = StateGraph(MessagesState)

    graph.add_node("agent", call_llm)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})

    graph.add_edge("tools", "agent")

    return graph.compile()
