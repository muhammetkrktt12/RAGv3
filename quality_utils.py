"""Metin işleme, lexical reranking, alıntı doğrulama ve değerlendirme yardımcı fonksiyonları."""
import re
from lexical_similarity_test import (
    kelimelere_ayir,
    ucgramlar,
    lexical_similarity,
    rerank_candidates as hybrid_rerank
)

def tokens(text):
    """Metni küçük harfli kelime token'larına ayırır."""
    return re.findall(r"\w+", text.lower())

def lexical_score(query, text):
    """Sorgu ile metin arasındaki 3-gram lexical benzerlik skoru."""
    return lexical_similarity(query, text)

def rerank_candidates(query, candidates, top_n=3):
    """Aday dokümanları Türkçe 3-gram hibrit benzerliğe göre sıralar [(doc, score), ...]."""
    return hybrid_rerank(query, candidates, lexical_agirlik=0.4, top_n=top_n)

def validate_citations(answer, known_labels):
    """Yanıt içerisindeki [K1], [K2] biçimindeki atıfları doğrular."""
    found = re.findall(r"\[(K\d+)\]", answer)
    known_set = set(known_labels)
    invalid_citations = [f for f in found if f not in known_set]
    is_valid = len(found) > 0 and len(invalid_citations) == 0
    return {
        "valid": is_valid,
        "found": found,
        "invalid": invalid_citations
    }

def hit_at_k(docs, expected):
    """
    İlk k dokümanda beklenen kaynağın (source) veya sayfanın (page) olup olmadığını döner (1 veya 0).
    expected int veya liste ise sayfa numarası (page), string ise dosya adı (source) kontrol edilir.
    """
    if isinstance(expected, (int, list, tuple, set)):
        expected_pages = {expected} if isinstance(expected, int) else set(expected)
        return int(any(d.metadata.get("page") in expected_pages for d in docs))
    return int(any(d.metadata.get("source") == expected for d in docs))
