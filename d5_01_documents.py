"""5. gün / Lab 1: TXT ve PDF dokümanlarını yükleme + metadata yönetimi."""
import json
from pathlib import Path
import pypdf
from langchain_core.documents import Document
from config import ROOT, DATA_DIR

def get_metadata_map():
    meta_file = ROOT / "meta.json"
    if meta_file.exists():
        return json.loads(meta_file.read_text(encoding="utf-8"))
    return {}

def load_pdf_documents(pdf_name="fenbilimleri.pdf"):
    """data klasöründeki belirtilen PDF (veya tüm PDF'leri) sayfa sayfa okur."""
    metadata = get_metadata_map()
    docs = []
    
    if pdf_name:
        pdf_paths = [DATA_DIR / pdf_name] if (DATA_DIR / pdf_name).exists() else []
    else:
        pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

    for path in pdf_paths:
        if not path.exists():
            continue
        reader = pypdf.PdfReader(str(path))
        file_meta = metadata.get(path.name, {"birim": "Genel", "erisim": "genel"})
        
        for idx, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            meta = {
                "source": path.name,
                "page": idx,
                **file_meta
            }
            docs.append(Document(page_content=text, metadata=meta))
    return docs

def load_documents(include_pdf=True):
    """Hem txt hem de pdf dokümanlarını yükler."""
    metadata = get_metadata_map()
    docs = []
    
    # 1. TXT belgeleri yükle
    for path in sorted(DATA_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        meta = {"source": path.name, **metadata.get(path.name, {"birim": "Genel", "erisim": "genel"})}
        docs.append(Document(page_content=content, metadata=meta))
    
    # 2. PDF belgeleri yükle
    if include_pdf:
        pdf_docs = load_pdf_documents(pdf_name=None)
        docs.extend(pdf_docs)

    return docs

if __name__ == "__main__":
    docs = load_documents(include_pdf=True)
    print("Toplam Belge Sayısı (TXT + PDF Sayfaları):", len(docs))
    for d in docs:
        page_info = f" (Sayfa {d.metadata['page']})" if "page" in d.metadata else ""
        print(f"- {d.metadata['source']}{page_info} | {d.metadata['birim']} | {len(d.page_content)} karakter")
