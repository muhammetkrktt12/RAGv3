"""5. gün / Lab 2: chunking + overlap + metadata koruma (PDF ve TXT desteği + Token Sayımı)."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from d5_01_documents import load_documents, load_pdf_documents
from tokenizer_utils import get_gemma_tokenizer

def make_chunks(docs, chunk_size=420, chunk_overlap=80):
    """Verilen belge listesini belirtilen boyut ve örtüşmeyle (overlap) chunk'lara böler."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    counts = {}
    for doc in chunks:
        source = doc.metadata.get("source", "doc")
        index = counts.get(source, 0)
        doc.metadata["chunk_id"] = f"{source}:{index:03d}"
        counts[source] = index + 1
    return chunks

def get_pdf_chunks(pdf_name="fenbilimleri.pdf", chunk_size=420, chunk_overlap=80):
    """Doğrudan data klasöründeki PDF dosyasını okuyup chunk'larını döner."""
    pdf_docs = load_pdf_documents(pdf_name=pdf_name)
    return make_chunks(pdf_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

if __name__ == "__main__":
    tokenizer = get_gemma_tokenizer()
    print("--- [PDF Chunking İşlemi: fenbilimleri.pdf] ---")
    pdf_chunks = get_pdf_chunks("fenbilimleri.pdf")
    print(f"Oluşturulan PDF Chunk Sayısı: {len(pdf_chunks)}\n")
    
    print("İlk 5 PDF Chunk örneği (Karakter & Token Sayımları):")
    for idx, c in enumerate(pdf_chunks[:5], 1):
        page = c.metadata.get("page", "?")
        token_count = len(tokenizer.encode(c.page_content).tokens)
        char_count = len(c.page_content)
        print(f"[{idx}] {c.metadata['chunk_id']} (Sayfa {page}) [{char_count} karakter | {token_count} token]:")
        print(f"    {repr(c.page_content[:100])}...\n")

    print("\n--- [Tüm Dokümanlar (TXT + PDF) Chunking] ---")
    all_chunks = make_chunks(load_documents(include_pdf=True))
    print(f"Toplam Doküman Chunk Sayısı: {len(all_chunks)}")
