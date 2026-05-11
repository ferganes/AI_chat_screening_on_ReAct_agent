import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from httpx_ws.transport import ASGIWebSocketTransport
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from main import app


@pytest_asyncio.fixture(scope="session")
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def ws_client():
    async with AsyncClient(
        transport=ASGIWebSocketTransport(app),
        base_url="http://test"
    ) as client:
        yield client

@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.ainvoke = AsyncMock()
    return agent


@pytest.fixture
def system_prompt():
    """Фикстура для системного промпта."""
    return "Ты полезный ассистент."


@pytest.fixture
def sample_chat_history():
    """Фикстура для примера истории чата."""
    return [
        HumanMessage(content="Привет"),
        AIMessage(content="Здравствуй!"),
    ]
