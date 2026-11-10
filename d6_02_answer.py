"""6. gün / Lab 2: Kaynaklı cevap; doküman talimat değil veridir."""
import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from config import CHAT_MODEL, OLLAMA_URL
from d6_01_rerank import retrieve_and_rerank
from data import *
"""
SYSTEM = (
    "Sen dokümanlara dayalı yanıt veren bir eğitim asistanısın. "
    "Kullanıcının sorusunu yalnızca iletilen KAYNAK METİNLERİNE dayanarak Türkçe olarak net ve açıklayıcı şekilde cevapla. "
    "Cevabındaki bilgileri destekleyen [K1], [K2] gibi kaynak etiketlerini mutlaka kullan. "
    "Yalnızca kaynaklarda konuyla ilgili hiçbir bilgi kesinlikle yoksa 'Bu bilgi dokümanlarda bulunamadı.' de."
)"""

def build_context(ranked):
    parts, provenance = [], {}
    for i, (doc, score) in enumerate(ranked, 1):
        label = f"K{i}"
        source = doc.metadata.get("source", "Bilinmiyor")
        chunk_id = doc.metadata.get("chunk_id", "-")
        page = doc.metadata.get("page", None)
        provenance[label] = {
            "source": source,
            "chunk_id": chunk_id,
            "page": page
        }
        page_str = f" page={page}" if page else ""
        parts.append(f"[{label}] source={source}{page_str} chunk_id={chunk_id}\n{doc.page_content}")
    return "\n\n".join(parts), provenance

def answer_question(question):
    ranked = retrieve_and_rerank(question)
    context, provenance = build_context(ranked)
    prompt = ChatPromptTemplate.from_messages([

        ("human", "SORU: {question}\n\nKAYNAK METİNLERİ:\n{context}"),
    ])
    llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_URL, temperature=0.3, num_predict=300)
    messages = prompt.format_messages(question=question, context=context)
    text = llm.invoke(messages).content
    return text, provenance

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Müşteri adı orkun olan veriyi bana getirir misin ?"

    print("=" * 60)
    print(f"SORU: {question}")
    print("=" * 60)
    print("Yanıt üretiliyor...\n")
    
    answer, sources = answer_question(question)
    
    print("--- MODEL YANITI ---")
    print(answer)
    print("\n--- KULLANILAN KAYNAKLAR (PROVENANCE) ---")
    for label, info in sources.items():
        page_info = f" (Sayfa {info['page']})" if info.get("page") else ""
        print(f"[{label}] -> {info['source']}{page_info} | {info['chunk_id']}")
    print("=" * 60)
