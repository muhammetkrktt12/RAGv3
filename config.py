"""Tek merkezden kullanılan yerel ayarlar. Sınıfta model adını buradan değiştirin."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
# Vektörleri data klasörüne chroma datası olarak kaydetmek için:
DB_DIR = DATA_DIR / "chroma_db"
COLLECTION = "kampus_belgeleri_v1"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:1.5b")
