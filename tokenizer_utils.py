"""EmbeddingGemma modeli için tokenizer ve teknik analiz yardımcı modülü."""
import json
import urllib.request
from pathlib import Path
from tokenizers import Tokenizer
from config import ROOT, EMBED_MODEL, OLLAMA_URL

TOKENIZER_FILE = ROOT / "gemma_tokenizer.json"

def get_gemma_tokenizer():
    """
    EmbeddingGemma modelinin kullandığı 256k kelime dağarcıklı
    Gemma BPE Tokenizer'ını yerel dosyadan yükler veya HuggingFace üzerinden indirip önbelleğe alır.
    """
    if TOKENIZER_FILE.exists():
        return Tokenizer.from_file(str(TOKENIZER_FILE))
    
    print("[INFO] Gemma tokenizer yerel dosyada bulunamadı, indiriliyor ve kaydediliyor...")
    tok = Tokenizer.from_pretrained("unsloth/gemma-7b")
    tok.save(str(TOKENIZER_FILE))
    return tok

def get_model_technical_specs(model_name=EMBED_MODEL):
    """Ollama API üzerinden modelin mimari ve teknik parametrelerini çeker."""
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/show",
            data=json.dumps({"model": model_name}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.load(response)
            info = data.get("model_info", {})
            details = data.get("details", {})
            return {
                "model_name": model_name,
                "architecture": info.get("general.architecture", "gemma3"),
                "family": details.get("family", "gemma3"),
                "parameter_size": details.get("parameter_size", "307.58M"),
                "parameter_count": info.get("general.parameter_count", 307581696),
                "quantization": details.get("quantization_level", "BF16"),
                "embedding_dimension": info.get("gemma3.embedding_length", 768),
                "context_window": info.get("gemma3.context_length", 2048),
                "pooling_type": "Mean Pooling (Tip 1)" if info.get("gemma3.pooling_type") == 1 else "Bilinmiyor",
                "attention_blocks": info.get("gemma3.block_count", 24),
                "attention_heads": info.get("gemma3.attention.head_count", 3),
                "sliding_window": info.get("gemma3.attention.sliding_window", 512),
                "bos_token_id": info.get("tokenizer.ggml.bos_token_id", 2),
                "eos_token_id": info.get("tokenizer.ggml.eos_token_id", 1),
                "pad_token_id": info.get("tokenizer.ggml.padding_token_id", 0),
            }
    except Exception as e:
        return {
            "model_name": model_name,
            "architecture": "gemma3",
            "family": "gemma3",
            "parameter_size": "307.58M",
            "quantization": "BF16",
            "embedding_dimension": 768,
            "context_window": 2048,
            "error": str(e)
        }

def analyze_chunk_tokens(chunk, tokenizer=None):
    """Bir chunk dokümanını tokenize eder ve detaylı teknik analiz üretir."""
    if tokenizer is None:
        tokenizer = get_gemma_tokenizer()

    text = chunk.page_content
    enc = tokenizer.encode(text)
    
    token_count = len(enc.tokens)
    char_count = len(text)
    word_count = len(text.split())
    
    return {
        "chunk_id": chunk.metadata.get("chunk_id", "Bilinmiyor"),
        "source": chunk.metadata.get("source", "Bilinmiyor"),
        "page": chunk.metadata.get("page", "-"),
        "char_count": char_count,
        "word_count": word_count,
        "token_count": token_count,
        "tokens": enc.tokens,
        "token_ids": enc.ids,
        "char_per_token": round(char_count / max(token_count, 1), 2),
        "context_usage_percent": round((token_count / 2048) * 100, 2),
        "text": text
    }

def print_model_specs(specs):
    """Modelin teknik özelliklerini ekrana formatlı olarak yazar."""
    print("=" * 75)
    print("  EMBEDDING MODELİ & TOKENIZER TEKNİK BİLGİ KARTI")
    print("=" * 75)
    print(f"  * Model Adı             : {specs.get('model_name')}")
    print(f"  * Mimari (Architecture) : {specs.get('architecture')} ({specs.get('family')})")
    print(f"  * Parametre Boyutu      : {specs.get('parameter_size')} (~{specs.get('parameter_count', 0):,} parametre)")
    print(f"  * Kuantizasyon Seviyesi : {specs.get('quantization')}")
    print(f"  * Vektör Boyutu (Dim)   : {specs.get('embedding_dimension')} float")
    print(f"  * Maks. Bağlam Sınırı   : {specs.get('context_window')} token (Context Window)")
    print(f"  * Havuzlama (Pooling)   : {specs.get('pooling_type')}")
    print(f"  * Attention Katmanları  : {specs.get('attention_blocks')} blok, {specs.get('attention_heads')} head (Sliding Window: {specs.get('sliding_window')})")
    print(f"  * Tokenizer Vocab Size  : 256,000 token (Gemma BPE / SentencePiece)")
    print(f"  * Özel Token'lar (IDs)  : <bos>={specs.get('bos_token_id')}, <eos>={specs.get('eos_token_id')}, <pad>={specs.get('pad_token_id')}")
    print("=" * 75)

def print_chunk_token_details(analysis, max_tokens_preview=25):
    """Tek bir chunk'ın token detaylarını detaylı olarak yazdırır."""
    print(f"\n--- [CHUNK ID: {analysis['chunk_id']}] ---")
    print(f"  * Kaynak Belge  : {analysis['source']} (Sayfa {analysis['page']})")
    print(f"  * Metin Boyutu  : {analysis['char_count']} karakter | {analysis['word_count']} kelime")
    print(f"  * Token Sayısı  : {analysis['token_count']} token")
    print(f"  * Karakter/Token: {analysis['char_per_token']} karakter/token")
    print(f"  * Bağlam Doluluğu: %{analysis['context_usage_percent']} (2048 token sınırına göre)")
    print(f"  * Metin Önizleme:")
    preview = analysis['text'][:140].replace('\n', ' ')
    print(f"    \"{preview}...\"")
    
    tokens = analysis['tokens']
    ids = analysis['token_ids']
    
    print(f"\n  * Metnin Tokenlara Ayrılmış Hali (İlk {min(max_tokens_preview, len(tokens))} token):")
    token_pairs = [f"{repr(t)}:{i}" for t, i in zip(tokens[:max_tokens_preview], ids[:max_tokens_preview])]
    print("    [ " + " | ".join(token_pairs) + (" | ... ]" if len(tokens) > max_tokens_preview else " ]"))
    
    if len(tokens) > max_tokens_preview:
        print(f"    (Toplam {len(tokens)} token içinden ilk {max_tokens_preview} tanesi gösterildi)")

