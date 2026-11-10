"""5. gün / Lab 3: Ollama embedding + Tokenizer analizi + Vektörleri Chroma'ya kaydetme."""
import shutil
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import EMBED_MODEL, OLLAMA_URL, DB_DIR, COLLECTION, DATA_DIR
from d5_02_chunks import get_pdf_chunks, make_chunks
from d5_01_documents import load_documents
from tokenizer_utils import (
    get_gemma_tokenizer,
    get_model_technical_specs,
    analyze_chunk_tokens,
    print_model_specs,
    print_chunk_token_details
)

def embedding_client():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

def embed_and_save_to_chroma(chunks=None, persist_dir=None, reset=True, batch_size=50):
    """
    Chunk'ları Ollama ile embedding edip vektörlerine ayırır
    ve data klasörüne Chroma veritabanı olarak kaydeder.
    """
    target_dir = Path(persist_dir) if persist_dir else DB_DIR

    if reset and target_dir.exists():
        try:
            shutil.rmtree(target_dir)
            print(f"\nMevcut Chroma veritabanı temizleniyor: {target_dir}")
        except PermissionError:
            print(f"\n[BİLGİ] {target_dir / 'chroma.sqlite3'} dosyası bir SQLite görüntüleyicisinde açık olduğu için klasör silinmedi; mevcut koleksiyon üzerine yazılacak.")

    if chunks is None:
        # data klasöründeki PDF dosyasının chunk'larını al
        chunks = get_pdf_chunks("fenbilimleri.pdf")

    total = len(chunks)
    print(f"\nToplam {total} chunk vektörleştiriliyor ({EMBED_MODEL}) ve Chroma'ya kaydediliyor...")
    
    ids = [d.metadata["chunk_id"] for d in chunks]
    emb = embedding_client()
    
    store = None
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_docs = chunks[start:end]
        batch_ids = ids[start:end]
        print(f"  -> [{start+1}-{end} / {total}] chunk embedding ediliyor...")
        if store is None:
            store = Chroma.from_documents(
                documents=batch_docs,
                ids=batch_ids,
                embedding=emb,
                collection_name=COLLECTION,
                persist_directory=str(target_dir),
            )
        else:
            store.add_documents(documents=batch_docs, ids=batch_ids)

    return store, total

if __name__ == "__main__":
    # 1. Model ve Tokenizer Teknik Özelliklerini Yazdır
    specs = get_model_technical_specs(EMBED_MODEL)
    print_model_specs(specs)

    # 2. Embedding istemcisi kontrolü
    emb = embedding_client()
    sample_text = "Maddenin halleri nelerdir?"
    vector = emb.embed_query(sample_text)
    print(f"Örnek Sorgu: \"{sample_text}\"")
    print(f"  * Üretilen Vektör Boyutu: {len(vector)} boyut")
    print(f"  * İlk 5 Vektör Değeri   : {[round(x, 4) for x in vector[:5]]}\n")

    # 3. PDF chunk'larını al ve Tokenizer Analizi Yap
    print("=" * 75)
    print("  CHUNK BAZLI TOKENIZER DETAYLARI VE METİN BÖLÜNME ANALİZİ")
    print("=" * 75)
    
    tokenizer = get_gemma_tokenizer()
    chunks = get_pdf_chunks("fenbilimleri.pdf")
    
    # İlk 3 chunk için derinlemesine teknik detayları ve metnin tokenlara ayrılmasını yazdır
    for c in chunks[:3]:
        analysis = analyze_chunk_tokens(c, tokenizer)
        print_chunk_token_details(analysis, max_tokens_preview=25)

    # Genel veri seti seviyesinde token istatistikleri
    print("\n" + "=" * 75)
    print("  GENEL DOKÜMAN TOKEN İSTATİSTİKLERİ")
    print("=" * 75)
    all_token_counts = [len(tokenizer.encode(c.page_content).tokens) for c in chunks]
    total_tokens = sum(all_token_counts)
    avg_tokens = round(total_tokens / len(chunks), 1)
    min_tokens = min(all_token_counts)
    max_tokens = max(all_token_counts)
    max_context = specs.get("context_window", 2048)

    print(f"  * Toplam Chunk Sayısı           : {len(chunks)}")
    print(f"  * Toplam Üretilen Token Sayısı  : {total_tokens:,} token")
    print(f"  * Chunk Başına Ortalama Token   : {avg_tokens} token")
    print(f"  * En Küçük Chunk Token Sayısı   : {min_tokens} token")
    print(f"  * En Büyük Chunk Token Sayısı   : {max_tokens} token")
    print(f"  * Maks. Model Bağlamı Uygunluğu : {max_tokens} / {max_context} token (Bağlam taşması YOK - Güvenli)")
    print("=" * 75)

    # 4. Chunk'ları Chroma'ya kaydet
    store, n = embed_and_save_to_chroma(chunks=chunks, reset=True)
    print(f"\nBaşarıyla kaydedilen chunk sayısı : {n}")
    print(f"Chroma Veritabanı Yolu            : {DB_DIR}")
    print(f"Koleksiyondaki toplam kayıt sayısı: {len(store.get()['ids'])}")
    
    # 5. Doğrulama araması
    test_q = "Maddelerin nitelikleri ve özellikleri nelerdir?"
    results = store.similarity_search(test_q, k=2)
    print(f"\nTest Sorgusu: '{test_q}'")
    for i, res in enumerate(results, 1):
        page = res.metadata.get("page", "?")
        print(f"Sonuç {i}: {res.metadata.get('chunk_id')} (Sayfa {page}) -> {repr(res.page_content[:110])}...")
