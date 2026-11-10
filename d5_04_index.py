"""5. gün / Lab 4: Diskte saklanan Chroma vektör indeksi (PDF + TXT)."""
import shutil
from langchain_chroma import Chroma
from config import DB_DIR, COLLECTION, ROOT, DATA_DIR
from d5_01_documents import load_documents
from d5_02_chunks import make_chunks
from d5_03_embeddings import embedding_client

def build_index(reset=False):
    if reset and DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    docs = make_chunks(load_documents(include_pdf=True))
    ids = [d.metadata["chunk_id"] for d in docs]
    store = Chroma.from_documents(
        documents=docs,
        ids=ids,
        embedding=embedding_client(),
        collection_name=COLLECTION,
        persist_directory=str(DB_DIR),
    )
    return store, len(docs)

if __name__ == "__main__":
    db, n = build_index(reset=True)
    print("Kaydedilen chunk:", n)
    print("Chroma dizini:", DB_DIR)
    print("Koleksiyondaki kayıt:", len(db.get()["ids"]))
