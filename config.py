from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from tools import approve_user,reject_user,search_description_in_okved


# Константы llm & embeddings
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "qwen3.5:9b"  # qwen2.5:14b  qwen2.5:3b gemma3:12b llama3.1:8b  mistral-small qwen3.5:9b


EMBEDDINGS = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

LLM = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    num_ctx=2048,
    num_predict=256,
    # keep_alive="2h"
)

# LLM_WITH_BIND_TOOLS = LLM.bind_tools(
#     tools=[approve_user, reject_user, search_description_in_okved],
#     tool_choice="required"
# )

# Константы векторной базы данных
PERSIST_DIR = "./chroma_db"
COLLECTION = "okved"

# Константы сплиттера чанков
CHUNK_SIZE = 500
CHUNK_OVERLAP = 64
SEPARATORS = ["\n\n", "\n", ". ", " "]

# Количество релевантных документов в ретривере
RETRIEVER_SEARCH_K = 3

# сколько сообщений помнит агент
MAX_AGENT_ITERATIONS = 10
AGENT_TIMEOUT = 60
MEMORY_WINDOW_K = 5

# Количество хранимых сообщений чата
MAX_CHAT_HISTORY = 5

# Таймаут неактивности вебсокета
TIMEOUT = 900

# Размер сообщения
USER_MESSAGE_SIZE = 2000

