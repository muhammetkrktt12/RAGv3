"""6. gün / Lab 1: Retrieve(k=8) -> 3-gram Türkçe Lexical Rerank(top_n=3)."""
import sys
from d5_05_search import search
from lexical_similarity_test import rerank_candidates as lexical_rerank

def retrieve_and_rerank(question, k=8, top_n=6, lexical_weight=0.4):
    """
    Chroma'dan en benzer k dokümanı alır ve lexical_similarity_test modülü ile
    Türkçe 3-gram hibrit yeniden sıralama uygular.
    """
    candidates = search(question, k=k)
    return lexical_rerank(question, candidates, lexical_agirlik=lexical_weight, top_n=top_n)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "Maddelerin nitelikleri ve özellikleri nelerdir?"

    print(f"--- [HİBRİT YENİDEN SIRALAMA (Semantic + 3-gram Lexical)] ---")
    print(f"Sorgu: {q}\n")
    ranked = retrieve_and_rerank(q, k=8, top_n=3)
    for i, (doc, score) in enumerate(ranked, 1):
        page = doc.metadata.get("page", "-")
        chunk_id = doc.metadata.get("chunk_id", "-")
        source = doc.metadata.get("source", "?")
        print(f"[{i}] {source} (Sayfa {page}) | {chunk_id} | Hibrit Skor: {round(score, 3)}")
        print(f"    {repr(doc.page_content[:120])}...\n")
