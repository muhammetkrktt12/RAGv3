# BTK Akademi — Gün 5 ve 6 RAG Atölyesi (Kampüs Asistanı)

**Amaç:** LangChain + yerel Ollama + ChromaDB ile 5. gün bilgi arama altyapısı, 6. gün kaynaklı RAG oluşturmak. Tüm dokümanlar eğitim amacıyla kurgulanmıştır. Gerçek kişisel veri kullanmayın.

## Ders öncesi zorunlu kurulum (Windows PowerShell)

1. Python **3.11/3.12**, Ollama uygulaması ve VS Code yüklü olsun.
2. Proje klasöründe terminal açın:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull qwen2.5:1.5b
ollama pull embeddinggemma
ollama list
python 00_kurulum_kontrol.py
```

PowerShell yürütme politikası engellerse aktivasyon yerine `.\.venv\Scripts\python.exe` komutunu kullanın. `py -3.11` bulunamazsa yüklü Python sürümünü doğrulayın. Mac/Linux: `python3 -m venv .venv`, `source .venv/bin/activate`.

**Not:** Model indirmelerini kesinlikle ilk derste tüm sınıfa aynı anda yaptırmayın; ders öncesinde tamamlatın. `ollama serve` komutunu yalnızca servis zaten çalışmıyorsa çalıştırın. Ollama 11434 portunu internete açmayın.

## GÜN 5: önerilen komut sırası

```bash
python 00_kurulum_kontrol.py
python d5_01_documents.py
python d5_02_chunks.py
python d5_03_embeddings.py
python d5_04_index.py
python d5_05_search.py
```

**GÜN 6:**

```bash
python d6_01_rerank.py
python d6_02_answer.py
python d6_03_validate.py
python d6_04_eval.py
python d6_05_guvenlik.py
```

`python -m unittest -v test_logic.py` testleri internet/model olmadan çalışır. Öğrenci sürümünde day6 rerank TODO kodu tamamlanana kadar bazı testler bilerek başarısızdır (red → green). `d6_05_guvenlik.py` model çağrısı yapmadan güven sınırını gösterir; sınıf isterse yanıt üretim koduyla manuel red-team testi yapılır.

## Öğrenci dosyaları

Çalıştırılabilir dosyaların içinde öğretici `TODO` boşlukları vardır. Takıldığınızda eğitmenin referans çözümünden bir önceki aşamanın tamamlanmış kodunu alın. Eğitmen çözümünü baştan kopyalamayın.

## Doğrulama / kapsam sınırlamaları

- `similarity_search` sıralaması, embedding modelinin Türkçe performansına ve chunk boyutuna göre değişir.
- Lexical reranking gerçek bir ikinci sıralama sinyalidir ama neural cross-encoder değildir; basit bir öğretim yaklaşımıdır.
- `[K1]` biçimindeki bir atfın geçerli olması yanıtın gerçekten doğru olduğunu **kanıtlamaz**. İnsan değerlendirmesi / kanıt eşleştirme gerekir.
- Prompt injection savunması yalnızca prompt ile garanti edilmez. Kurumsal sistemde erişim kontrolü kaynak alma işleminden önce zorlanmalıdır.
- LangChain/ChromaDB paketleri bu dosyaların üretildiği ortamda kurulup uçtan uca çalıştırılamadı; ilk ders öncesi kendi bilgisayarınızda preflight ve 5–8 soru doğrulaması zorunludur. `requirements-lock.txt` gerçek kurulumdan sonra oluşturulmalıdır.
- `chroma_kampus` klasörü yalnızca bu eğitim verisi içindir. `d5_04_index.py` tekrar çalıştırıldığında bu klasör silinip indeks yeniden oluşturulur. Gerçek veride asla kullanmayın.

## Çevrim dışı yedek

```bash
python offline_yedek.py
```

Bu script **RAG değil**, yalnızca standart Python ile anahtar kelime eşleşmesi gösterir. Ollama/Chroma arızalanırsa kavram tartışması ve kıyaslama için kullanın; çalışan RAG gibi sunmayın.

## Resmî dokümanlar
- https://reference.langchain.com/python/langchain-ollama
- https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma
- https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter
- https://docs.ollama.com/api
