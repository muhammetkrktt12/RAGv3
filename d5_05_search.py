"""5. gün / Lab 5: anlamsal arama ve metadata filtreleme."""
import sys
from langchain_chroma import Chroma
from config import DB_DIR, COLLECTION
from d5_03_embeddings import embedding_client

def get_store():
    if not DB_DIR.exists():
        raise FileNotFoundError(f"Chroma dizini bulunamadı ({DB_DIR}). Önce python d5_03_embeddings.py veya d5_04_index.py çalıştırın.")
    return Chroma(collection_name=COLLECTION, embedding_function=embedding_client(), persist_directory=str(DB_DIR))

def search(question, k=4, unit=None):
    filter_value = {"birim": unit} if unit else None
    return get_store().similarity_search(query=question, k=k, filter=filter_value)

if __name__ == "__main__":
    # Komut satırından argüman verilmişse onu al, verilmemişse örnek sorguyu çalıştır
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "İlk okuma yazmada dijital araçların ve oyunların etkisi nedir?"

    print(f"--- [ANLAMSAL ARAMA (Semantic Search)] ---")
    print(f"Soru: {question}\n")
    results = search(question, k=4)
    for idx, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Bilinmiyor")
        page = doc.metadata.get("page", "-")
        chunk_id = doc.metadata.get("chunk_id", "-")
        print(f"[{idx}] Kaynak: {source} (Sayfa: {page}) | ID: {chunk_id}")
        print(f"    İçerik Özeti: {repr(doc.page_content[:130])}...\n")


