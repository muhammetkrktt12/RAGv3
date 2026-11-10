"""LangChain ve Ollama olmadan koşan testler: python -m unittest -v test_logic.py"""
import unittest
from types import SimpleNamespace
from pathlib import Path
from config import ROOT
from quality_utils import tokens, lexical_score, rerank_candidates, validate_citations, hit_at_k

def doc(content, name):
    return SimpleNamespace(page_content=content, metadata={"source":name})

class Tests(unittest.TestCase):
    def test_tokens_tr(self): self.assertIn("kütüphane", tokens("Kütüphane açık mı?"))
    def test_lexical_score(self): self.assertGreater(lexical_score("kütüphane saat", "kütüphane saat 22"),0)
    def test_rerank_changes_order(self):
        items=[doc("Sınav takvimi", "sinavlar.txt"), doc("Kütüphane saat 22", "kutuphane.txt")]
        self.assertEqual(rerank_candidates("kütüphane saat",items)[0][0].metadata["source"],"kutuphane.txt")
    def test_known_citations(self): self.assertTrue(validate_citations("Açık [K1]",["K1"])["valid"])
    def test_unknown_citations(self): self.assertIn("K9",validate_citations("Açık [K9]",["K1"])["invalid"])
    def test_missing_citations(self): self.assertFalse(validate_citations("Söylendi.",["K1"])["valid"])
    def test_hit_at_k(self): self.assertEqual(hit_at_k([doc("metin","kutuphane.txt")],"kutuphane.txt"),1)
    def test_data_count(self): self.assertGreaterEqual(len(list((ROOT/"data").glob("*.txt"))),8)

if __name__ == "__main__": unittest.main()
