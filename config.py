from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

# Константы llm & embeddings
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:14b"  # qwen2.5:14b  qwen2.5:3b


EMBEDDINGS = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

LLM = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    num_ctx=2000
)

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

