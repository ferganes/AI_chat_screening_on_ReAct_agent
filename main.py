from fastapi import Request
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import asyncio

import rag
from config import TIMEOUT, USER_MESSAGE_SIZE, LLM
from database import connect_database
from tools import set_vectorstore
from agent import create_agent
from websocket_handler import agent_process_message


templates = Jinja2Templates(directory="templates")


def _check_yandex_connection():
    """Проверка доступности YandexGPT"""
    try:
        LLM.invoke("ping")
        print("[STARTUP] YandexGPT доступен")
    except Exception as e:
        print(f"[WARNING] Не удалось подключиться к YandexGPT: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    print("Starting up...")

    # Проверяем LLM в отдельном потоке
    await asyncio.to_thread(_check_yandex_connection)

    # Подключаем/создаём БД в отдельном потоке
    vectorstore = await asyncio.to_thread(connect_database)
    set_vectorstore(vectorstore)

    # Инициализация RAG
    await asyncio.to_thread(rag.start_rag, vectorstore)

    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def create_chat_page(request: Request):
    """Главная страница чата"""
    return templates.TemplateResponse(request, "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket с таймаутом 15 минут"""
    await websocket.accept()

    chat_history = []
    agent = create_agent()
    print(f"[CONSOLE] Создан ReAct агент (YandexGPT) для нового чата")

    try:
        while True:
            user_message = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=TIMEOUT
            )

            if len(user_message) > USER_MESSAGE_SIZE:
                await websocket.send_text('Превышен лимит длины текстового сообщения')
            else:
                answer, chat_history = await agent_process_message(
                    agent, user_message, chat_history
                )
                await websocket.send_text(str(answer))

    except asyncio.TimeoutError:
        print(f"[CONSOLE] Таймаут WebSocket: {TIMEOUT/60:.0f} минут без активности")
        await websocket.send_text(f'Сеанс неактивен {TIMEOUT/60:.0f} минут и будет закрыт')
        await websocket.close(code=1000, reason="Timeout: inactivity")

    except WebSocketDisconnect:
        print("[CONSOLE] Клиент отключился")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=900
    )
