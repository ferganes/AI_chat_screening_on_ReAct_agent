from fastapi import Request
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import asyncio

import check_llm
import rag
from config import LLM, TIMEOUT, USER_MESSAGE_SIZE
from database import connect_database
from tools import set_vectorstore
from agent import create_agent
from websocket_handler import agent_process_message

templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""

    # Startup
    print("Starting up...")

    # Проверяем доступность llm
    check_llm.check_ollama()

    # Подключаем БД
    vectorstore = connect_database()
    set_vectorstore(vectorstore)

    # Инициализация RAG
    rag.start_rag(vectorstore)

    yield

    print("Shutting down...")


# Создаем приложение
app = FastAPI(lifespan=lifespan)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Маунт статики
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def create_chat_page(request: Request):
    """Главная страница чата"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket с таймаутом 15 минут"""
    await websocket.accept()

    chat_history = []
    agent = create_agent(LLM)
    print(f"[CONSOLE] Создан ReAct агент для нового чата")

    try:
        while True:
            # Ожидание сообщения с таймаутом сеанса
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
        print(f"[CONSOLE] Таймаут WebSocket: {TIMEOUT} минут без активности")
        await websocket.send_text(f'Сеанс неактивен {TIMEOUT/60} минут и будет закрыт')
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
