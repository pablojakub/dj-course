from tokenizers import Tokenizer
from corpora import get_corpus_file
import sys
from pathlib import Path
import os

TOKENIZERS = {
    "nkjp": "tokenizers/tokenizer-nkjp.json",
    "all": "tokenizers/tokenizer-all-corpora.json",
    "all64": "tokenizers/tokenizer-all-corpora-64.json",
    "tadek": "tokenizers/tokenizer-pan-tadeusz.json",
    "wolne": "tokenizers/tokenizer-wolnelektury.json",
    "bielik-v1": "tokenizers/bielik-v1-tokenizer.json",
    "bielik-v2": "tokenizers/bielik-v2-tokenizer.json",
    "bielik-v3": "tokenizers/bielik-v3-tokenizer.json",
}

# Wczytaj tekst źródłowy raz
source_txt = ""
corpus_files = get_corpus_file("MINI", "*.txt")
if not corpus_files:
    print("Nie znaleziono pliku korpusu: pan-tadeusz-ksiega-*.txt", file=sys.stderr)
    sys.exit(1)

with open(corpus_files[0], 'r', encoding='utf-8') as f:
    source_txt = f.read()

print(f"Długość tekstu źródłowego: {len(source_txt)} znaków\n")
print("="*60)

# Przetwórz wszystkie tokenizery
results = []

for tokenizer_name, tokenizer_path in TOKENIZERS.items():
    # Sprawdź czy plik tokenizera istnieje
    if not Path(tokenizer_path).exists():
        print(f"⚠️  Tokenizer '{tokenizer_name}' nie istnieje: {tokenizer_path}")
        print("-"*60)
        continue
    
    try:
        # Wczytaj tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Zakoduj tekst
        encoded = tokenizer.encode(source_txt)
        
        # Zapisz wynik do pliku
        file_name = f"logs/tokenized-Pan-Tadeusz-{tokenizer_name}.log"
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(f"Tokenizer: {tokenizer_name}\n")
            f.write(f"Ścieżka: {tokenizer_path}\n")
            f.write(f"Liczba tokenów: {len(encoded.ids)}\n")
            f.write(f"Długość tekstu: {len(source_txt)} znaków\n")
            f.write(f"Kompresja: {len(source_txt) / len(encoded.ids):.2f} znaków/token\n")
        
        # Wydrukuj wynik
        print(f"✓ Tokenizer: {tokenizer_name}")
        print(f"  Liczba tokenów: {len(encoded.ids)}")
        print(f"  Kompresja: {len(source_txt) / len(encoded.ids):.2f} znaków/token")
        print(f"  Zapisano do: {file_name}")
        print("-"*60)
        
        # Zapisz do listy wyników
        results.append({
            'name': tokenizer_name,
            'tokens': len(encoded.ids),
            'compression': len(source_txt) / len(encoded.ids)
        })
        
    except Exception as e:
        print(f"❌ Błąd przy przetwarzaniu tokenizera '{tokenizer_name}': {e}")
        print("-"*60)

# Podsumowanie
if results:
    print("\n" + "="*60)
    print("PODSUMOWANIE:")
    print("="*60)
    
    # Sortuj według liczby tokenów (mniej = lepiej)
    results_sorted = sorted(results, key=lambda x: x['tokens'])
    
    for i, result in enumerate(results_sorted, 1):
        print(f"{i}. {result['name']:15} - {result['tokens']:6} tokenów ({result['compression']:.2f} znaków/token)")
    
    print("="*60)
    print(f"Najlepszy: {results_sorted[0]['name']} ({results_sorted[0]['tokens']} tokenów)")
    print(f"Najgorszy: {results_sorted[-1]['name']} ({results_sorted[-1]['tokens']} tokenów)")
    print(f"Różnica: {results_sorted[-1]['tokens'] - results_sorted[0]['tokens']} tokenów "
          f"({(results_sorted[-1]['tokens'] / results_sorted[0]['tokens'] - 1) * 100:.1f}% więcej)")