"""Mevcut d5_05_search.py ve chroma_db ile 3-gram Türkçe lexical reranking modülü."""
import re
from d5_05_search import search

def kelimelere_ayir(metin):
    """Türkçe I/İ harflerini koruyarak kelimelere ayır."""
    metin = metin.replace("İ", "i").replace("I", "ı").lower()
    return re.findall(r"[a-zçğıöşü0-9]+", metin)

def ucgramlar(metin):
    """Türkçe ek ve kök farklılıklarına toleranslı karakter 3-gramları (trigrams)."""
    gruplar = set()
    for kelime in kelimelere_ayir(metin):
        kelime = f"^{kelime}$"
        for i in range(len(kelime) - 2):
            gruplar.add(kelime[i:i + 3])
    return gruplar

def lexical_similarity(soru, metin):
    """Soru 3-gramlarından belgede bulunanların oranı: [0, 1]."""
    soru_gram = ucgramlar(soru)
    if not soru_gram:
        return 0.0
    belge_gram = ucgramlar(metin)
    return len(soru_gram & belge_gram) / len(soru_gram)

def ara_ve_sirala(soru, aday_sayisi=8, lexical_agirlik=0.4, sonuc_sayisi=3):
    """Chroma'dan arama yapar ve hibrit skorla yeniden sıralar: [(toplam, eski_sira, lexical_puan, belge), ...]"""
    if aday_sayisi < 1 or sonuc_sayisi < 1:
        raise ValueError("Aday ve sonuç sayısı pozitif olmalı")
    if not 0 <= lexical_agirlik <= 1:
        raise ValueError("Lexical ağırlık 0 ile 1 arasında olmalı")

    adaylar = search(soru, k=aday_sayisi)
    puanli = []
    for sira, belge in enumerate(adaylar, 1):
        anlamsal_sira_puani = 1 / sira
        lexical_puan = lexical_similarity(soru, belge.page_content)
        toplam = ((1 - lexical_agirlik) * anlamsal_sira_puani + lexical_agirlik * lexical_puan)
        puanli.append((toplam, sira, lexical_puan, belge))

    puanli.sort(key=lambda item: (-item[0], item[1]))
    return puanli[:sonuc_sayisi]

def rerank_candidates(soru, adaylar, lexical_agirlik=0.4, top_n=3):
    """
    d6_01_rerank, d6_02_answer ve d6_04_eval ile doğrudan entegre çalışacak fonksiyon.
    Dönüş formatı: [(belge, toplam_skor), ...]
    """
    if not adaylar:
        return []
    puanli = []
    for sira, belge in enumerate(adaylar, 1):
        anlamsal_sira_puani = 1 / sira
        lexical_puan = lexical_similarity(soru, belge.page_content)
        toplam = ((1 - lexical_agirlik) * anlamsal_sira_puani + lexical_agirlik * lexical_puan)
        puanli.append((belge, toplam))

    puanli.sort(key=lambda item: -item[1])
    return puanli[:top_n]

def main():
    soru = input("Soru : ").strip()
    soru = soru or "Maddenin halleri nelerdir, açıkla?"
    k_input = input("Chroma aday sayısı k [8]: ").strip()
    k = int(k_input) if k_input else 8
    agirlik_input = input("Lexical ağırlık 0-1 [0.4]: ").strip()
    agirlik = float(agirlik_input.replace(",", ".")) if agirlik_input else 0.4

    print("\n--- CHROMA'NIN İLK ANLAMSAL SIRALAMASI + LEXICAL PUANLARI ---")

    adaylar = search(soru, k=k)

    for i, belge in enumerate(adaylar, 1):
        chunk_id = belge.metadata.get("chunk_id", "-")
        page = belge.metadata.get("page", "-")
        source = belge.metadata.get("source", "?")

        lexical = lexical_similarity(soru, belge.page_content)

        print(
            f"{i:>2}. {source} (Sayfa {page}) | {chunk_id} "
            f"| Lexical={lexical:.3f}"
        )

    print("\n--- 3-GRAM LEXICAL & HİBRİT YENİDEN SIRALAMA ---")
    sirali = ara_ve_sirala(soru, aday_sayisi=k, lexical_agirlik=agirlik)
    for i, (toplam, eski_sira, lexical, belge) in enumerate(sirali, 1):
        chunk_id = belge.metadata.get("chunk_id", "-")
        page = belge.metadata.get("page", "-")
        source = belge.metadata.get("source", "?")
        print(f"{i}. {source} (Sayfa {page}) | {chunk_id} | Eski Sıra={eski_sira} | Lexical={lexical:.3f} | Birleşik={toplam:.3f}")
        print("   Metin:", belge.page_content[:150].replace("\n", " ") + "...\n")

if __name__ == "__main__":
    main()