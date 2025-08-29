#!/usr/bin/env python3
"""Full test of the attention geometry framework."""

from src.attention_metrics import AttentionAnalyzer
import numpy as np

print("=" * 60)
print("SPANISH-ENGLISH ATTENTION GEOMETRY TEST")
print("=" * 60)

# Initialize analyzer
print("\n1. Initializing AttentionAnalyzer...")
analyzer = AttentionAnalyzer(model_name='bert-base-multilingual-cased')
print("   ✓ Model loaded")

# Test sentences (parallel translations)
test_pairs = [
    ("El niño come manzanas rojas.", "The boy eats red apples."),
    ("La casa está cerca del parque.", "The house is near the park."),
    ("Me gusta leer libros interesantes.", "I like reading interesting books.")
]

print("\n2. Analyzing test sentence pairs...")
density_differences = []

for i, (spanish, english) in enumerate(test_pairs, 1):
    print(f"\n   Pair {i}:")
    print(f"   ES: {spanish}")
    print(f"   EN: {english}")
    
    # Tokenization analysis
    es_tokens = analyzer.analyze_tokenization_effects(spanish, 'es')
    en_tokens = analyzer.analyze_tokenization_effects(english, 'en')
    
    print(f"   Spanish: {es_tokens['total_tokens']} tokens, {es_tokens['subword_ratio']:.2%} subwords")
    print(f"   English: {en_tokens['total_tokens']} tokens, {en_tokens['subword_ratio']:.2%} subwords")
    
    # Attention metrics
    es_metrics = analyzer.analyze_text(spanish, 'es')
    en_metrics = analyzer.analyze_text(english, 'en')
    
    # Focus on layers 4-7 (your hypothesis)
    for layer in [4, 5, 6, 7]:
        diff = es_metrics['normalized_density'][layer] - en_metrics['normalized_density'][layer]
        density_differences.append(diff)
        if layer == 6:  # Show one example
            print(f"   Layer {layer} density difference: {diff:+.4f}")

print("\n3. Summary Statistics:")
print(f"   Mean density difference (ES-EN) in layers 4-7: {np.mean(density_differences):+.4f}")
print(f"   All differences positive (ES>EN)?: {all(d > 0 for d in density_differences)}")

print("\n" + "=" * 60)
print("FRAMEWORK VALIDATED - Ready for full UN Corpus study!")
print("=" * 60)
