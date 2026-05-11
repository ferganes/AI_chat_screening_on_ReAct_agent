import pytest
from httpx import AsyncClient
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from main import app


@pytest.mark.asyncio
async def test_chat_page_endpoint(client: AsyncClient):
    """Проверяем, что эндпоинт возвращает статус 200"""
    response = await client.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_page_content_type(client: AsyncClient):
    """Проверяем, что эндпоинт '/' возвращает text/html"""
    response = await client.get("/")
    assert "text/html" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_static_file_mounted(client: AsyncClient):
    """Проверяем маунт '/static', достаточно ответа 200 по наличию файла css"""
    response = await client.get("/static/style.css")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_websocket_connect():
    """Проверяем соединение с вебсокетом как ендпоинтом"""
    async with AsyncClient(
            transport=ASGIWebSocketTransport(app),
            base_url="http://test"
    ) as client:
        # открываем соединение и закрываем его,
        async with aconnect_ws("/ws", client) as websocket:
            pass
            await websocket.close()
