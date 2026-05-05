# ContextRevive — Central Configuration
# All AI runs locally via Ollama. No external API keys needed.

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
MAX_TOKENS_PER_SEGMENT = 512
MAX_MEMORY_SLOTS = 50
RELEVANCE_THRESHOLD = 0.65
TOP_K_MEMORIES = 5
DB_PATH = "./storage/contextrevive.db"
CHROMA_PATH = "./storage/chroma_db"
