import requests

from config import LLM_MODEL, EMBEDDING_MODEL


def check_ollama() -> bool:
    """Проверяет доступность локальной LLM"""
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        if response.status_code == 200:
            print(f"OK | Ollama работает (LLM: {LLM_MODEL}, Embedding: {EMBEDDING_MODEL})")
            return True
        return False
    except requests.RequestException:
        print("X | Ollama недоступна")
        return False
