"""6. gün / Lab 4: fenbilimleri.pdf için gerçek aramada Recall@3 + Rerank Top1 doğrulama ölçümü."""
from d5_05_search import search
from quality_utils import hit_at_k, rerank_candidates

# Fen Bilimleri 3. Sınıf Ders Kitabına ait soru ve kitabın ilgili sayfa numaraları
# Beklenen değer tek bir sayfa int (örn: 68) veya geçerli sayfalar listesi (örn: [12, 13]) olabilir.
CASES = [
    ("Laboratuvarda uyulması gereken güvenlik sembolleri nelerdir?", [12, 13]),
    ("Bilim insanlarının sahip olduğu ortak özellikler nelerdir?", [31, 37]),
    ("Duyu organlarımızın sağlığını korumak için nelere dikkat etmeliyiz?", [68, 69]),
    ("Kayaçlar, madenler ve minerallerin özellikleri nelerdir?", [88, 91, 92]),
    ("Fosil oluşumu nasıl gerçekleşir?", [98, 99, 100, 104]),
    ("Maddelerin nitelikleri nelerdir?", [111, 114, 136]),
    ("Maddenin katı, sıvı ve gaz hâlleri nelerdir?", [114, 115, 116]),
    ("Karışımlar hangi yöntemlerle birbirinden ayrılabilir?", [125, 126, 127]),
    ("Atıkların ayrıştırılması ve geri dönüşümün önemi nedir?", [132, 134, 140, 141]),
]

def format_expected_pages(expected):
    """Beklenen sayfa numarasını formatlı stringe çevirir."""
    if isinstance(expected, (list, tuple, set)):
        if len(expected) == 1:
            return f"Sayfa {list(expected)[0]}"
        return f"Sayfa {', '.join(map(str, sorted(expected)))}"
    return f"Sayfa {expected}"

if __name__ == "__main__":
    hits = 0
    top1 = 0
    total = len(CASES)

    print("=" * 80)
    print("  FEN BİLİMLERİ RAG DOĞRULAMA & DEĞERLENDİRME TESTİ (EVALUATION)")
    print("=" * 80)
    print(f"Toplam {total} test sorusu Chroma anlamsal arama ve 3-gram reranking ile test ediliyor...\n")

    for idx, (query, expected) in enumerate(CASES, 1):
        # 1. Chroma'dan ilk 8 adayı getir
        docs = search(query, k=8)
        
        # 2. 3-gram lexical similarity ile yeniden sırala (top 3)
        ranked = rerank_candidates(query, docs, top_n=3)
        
        # 3. Sıralamadaki sayfaları ve dokümanı al
        top_pages = [d.metadata.get("page", "?") for d, _ in ranked]
        top1_doc = ranked[0][0] if ranked else None
        top1_page = top1_doc.metadata.get("page", None) if top1_doc else None
        top1_chunk_id = top1_doc.metadata.get("chunk_id", "-") if top1_doc else "-"
        
        # 4. Doğrulama kontrolleri
        expected_set = {expected} if isinstance(expected, int) else set(expected)
        is_top1 = top1_page in expected_set
        is_hit3 = hit_at_k(docs[:3], expected) or any(p in expected_set for p in top_pages)
        
        hits += int(is_hit3)
        top1 += int(is_top1)
        
        # 5. Ekrana formatlı durum yazdır
        if is_top1:
            durum = "[OK / TOP-1 TAM İSABET]"
        elif is_hit3:
            durum = "[HIT@3 İSABET (İlk 3'te)]"
        else:
            durum = "[KONTROL EDİLMELİ]"

        print(f"[{idx}/{total}] {durum}")
        print(f"  Soru            : \"{query}\"")
        print(f"  Beklenen Sayfa  : {format_expected_pages(expected)}")
        print(f"  Bulunan Sayfa   : Sayfa {top1_page} (Top-1) | ID: {top1_chunk_id}")
        print(f"  İlk 3 Sıralama  : {top_pages}")
        if top1_doc:
            preview = top1_doc.page_content[:150].replace('\n', ' ')
            print(f"  Sayfa İçeriği   : \"{preview}...\"")
        print("-" * 80)

    # Özet Karne
    hit3_percent = round((hits / total) * 100, 1)
    top1_percent = round((top1 / total) * 100, 1)

    print("\n" + "=" * 80)
    print("  DEĞERLENDİRME SONUÇ RAPORU (METRİKLER)")
    print("=" * 80)
    print(f"  * Toplam Test Sorusu Sayısı : {total}")
    print(f"  * Hit@3 (İlk 3'te Bulma)    : {hits}/{total} (%{hit3_percent})")
    print(f"  * Rerank Top-1 Tam İsabet   : {top1}/{total} (%{top1_percent})")
    print("=" * 80)
    if top1_percent >= 80:
        print("  [SONUÇ]: Mükemmel! RAG arama ve reranking pipeline'ı soruları doğru sayfalara yönlendiriyor.")
    else:
        print("  [SONUÇ]: Bazı sorular için chunk overlap veya k katsayıları optimize edilebilir.")
    print("=" * 80 + "\n")
