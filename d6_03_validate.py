"""6. gün / Lab 3: Atıf etiketi doğrulama (olgusal doğruluk garantisi DEĞİL)."""
from d6_02_answer import answer_question
from quality_utils import validate_citations

if __name__ == "__main__":
    answer, provenance = answer_question("Kulüp kurmak için kaç imza gerekir?")
    validation = validate_citations(answer, provenance.keys())
    print("CEVAP:", answer)
    print("ATIF KONTROLÜ:", validation)
    if not validation["valid"]:
        print("UYARI: Kaynak etiketi yok/geçersiz. Cevabı otomatik onaylamayın.")
