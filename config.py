from langchain_openai import ChatOpenAI
from langchain_community.embeddings.yandex import YandexGPTEmbeddings


# ==================== YANDEX CLOUD CREDENTIALS ====================
YANDEX_API_KEY = "yandex api key"
YANDEX_FOLDER_ID = "<yandex folder id>"
YANDEX_LLM_MODEL = "aliceai-llm-flash/latest"


# ==================== LLM (OpenAI-compatible API) ====================
LLM = ChatOpenAI(
    model=f"gpt://{YANDEX_FOLDER_ID}/{YANDEX_LLM_MODEL}",
    api_key=YANDEX_API_KEY,
    base_url="https://ai.api.cloud.yandex.net/v1",
    default_headers={"OpenAI-Project": YANDEX_FOLDER_ID},
    temperature=0.0,  # ← 0 вместо 0.1 для строгого следования инструкциям
    max_tokens=2048,
    max_retries=2,
    timeout=60,
    model_kwargs={
        "parallel_tool_calls": False,
    }
)


# ==================== EMBEDDINGS (YandexGPT) ====================
EMBEDDINGS = YandexGPTEmbeddings(
    api_key=YANDEX_API_KEY,
    folder_id=YANDEX_FOLDER_ID,
)


# ==================== VECTOR DB & SPLITTER ====================
PERSIST_DIR = "./chroma_db"
COLLECTION = "okved"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 64
SEPARATORS = ["\n\n", "\n", ". ", " "]

RETRIEVER_SEARCH_K = 3


# ==================== AGENT & CHAT LIMITS ====================
MAX_AGENT_ITERATIONS = 10
AGENT_TIMEOUT = 60
MEMORY_WINDOW_K = 5

MAX_CHAT_HISTORY = 5

TIMEOUT = 900
USER_MESSAGE_SIZE = 2000
