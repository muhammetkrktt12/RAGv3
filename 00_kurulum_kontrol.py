"""Kurulum ön kontrolü: python 00_kurulum_kontrol.py"""
import importlib
import json
import urllib.request
from config import OLLAMA_URL, CHAT_MODEL, EMBED_MODEL

def main():
    package_names = ["langchain_core", "langchain_text_splitters", "langchain_ollama", "langchain_chroma", "chromadb"]
    print("[1] Python paketleri")
    missing = []
    for name in package_names:
        try:
            importlib.import_module(name)
            print("  OK", name)
        except ImportError as exc:
            print("  EKSIK", name, "=>", exc)
            missing.append(name)
    print("[2] Ollama API:", OLLAMA_URL + "/api/tags")
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=5) as response:
            tags = json.load(response).get("models", [])
        models = {x.get("name", "") for x in tags}
        print("  Kurulu modeller:", ", ".join(sorted(models)) or "YOK")
        for required in (CHAT_MODEL, EMBED_MODEL):
            installed = required in models or any(x.split(":")[0] == required for x in models)
            print("  ", "OK" if installed else "EKSIK", required)
    except Exception as exc:
        print("  HATA: Ollama bağlantısı kurulamadı:", exc)
        print("  Ollama uygulamasını açın veya terminalde ollama serve komutunu deneyin.")
    if missing:
        print("[ONERI] python -m pip install -r requirements.txt")

if __name__ == "__main__": main()
