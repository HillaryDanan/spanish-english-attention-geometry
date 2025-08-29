# Cross-Linguistic Attention Geometry in Transformer Models
## A Comparative Study of Spanish and English Processing Patterns

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

This repository contains code and analysis for investigating how orthographic transparency affects attention pattern geometry in multilingual transformer models. We test the hypothesis that transparent orthographies (Spanish) maintain denser associative patterns than opaque orthographies (English) in middle transformer layers.

## Core Hypothesis

**Orthographic transparency systematically affects attention pattern density in transformer models**, with measurable differences in graph-theoretic properties between Spanish and English text processing.

## Scientific Foundation

This work builds on established psycholinguistic and computational research:

- **Orthographic Depth Hypothesis** (Katz & Frost, 1992): Languages vary systematically in grapheme-phoneme transparency
- **Neural Processing Differences** (Paulesu et al., 2000, *Science*): Neuroimaging evidence for distinct processing strategies
- **Computational Attention Analysis** (Clark et al., 2019): Methods for interpreting transformer attention patterns

## Methodology

### Metrics

We employ three primary geometric metrics:

1. **Attention Density**: ρ = 2m/(n(n-1)) where m = edges above threshold
2. **Clustering Coefficient**: C = 3 × (triangles)/(connected triplets)  
3. **Hierarchy Index**: Gini coefficient of degree distribution

### Statistical Approach

- Wilcoxon signed-rank test for paired comparisons
- Benjamini-Hochberg FDR correction for multiple comparisons
- Effect size calculation using matched pairs correlation

### Data

- UN Parallel Corpus: 1000 sentence pairs (Spanish-English)
- Controlled for complexity using Flesch-Kincaid equivalence
- Length matching within ±10% token variance

## Repository Structure

```
spanish-english-attention-geometry/
├── src/
│   ├── attention_metrics.py    # Core metric calculations
│   ├── data_preprocessing.py   # Corpus preparation
│   └── statistical_tests.py    # Analysis functions
├── notebooks/
│   └── exploratory_analysis.ipynb
├── data/
│   ├── raw/                    # Original corpus (not tracked)
│   └── processed/               # Preprocessed pairs
├── results/
│   ├── figures/                 # Publication-ready plots
│   └── tables/                  # Statistical summaries
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/HillaryDanan/spanish-english-attention-geometry.git
cd spanish-english-attention-geometry
pip install -r requirements.txt
```

## Usage

```python
from src.attention_metrics import AttentionAnalyzer

analyzer = AttentionAnalyzer(model_name='bert-base-multilingual-cased')
spanish_metrics = analyzer.analyze_text(spanish_text, language='es')
english_metrics = analyzer.analyze_text(english_text, language='en')
```

## Expected Contributions

If our hypothesis is confirmed, this work will:

1. Provide first empirical evidence that orthographic properties create measurable computational differences in transformers
2. Establish methodology for cross-linguistic computational analysis
3. Bridge psycholinguistic theory with transformer interpretability

## Current Status

🚧 **Under Development** - Repository created August 2025

- [x] Study design and hypothesis formulation
- [x] Literature review and theoretical grounding
- [ ] Data collection and preprocessing
- [ ] Pilot study (N=35)
- [ ] Full analysis (N=1000)
- [ ] Statistical validation
- [ ] Manuscript preparation

## Future Directions

While the core study focuses on empirically testable claims, several speculative extensions merit future investigation:

- Information-theoretic constraints on attention patterns
- Potential connections to processing reversibility
- Geometric structure variations across language families

## Future Methodological Extensions

Based on initial review, several methodological enhancements are planned:

- **Tokenization Analysis**: Examine token-level patterns to isolate subword tokenization effects from orthographic transparency
- **Randomized Baselines**: Compare observed differences against randomized attention matrices to establish chance-level variations
- **Cross-Model Validation**: Extend beyond mBERT to XLM-R, mT5, and BLOOM for architectural robustness
- **Longitudinal Layer Analysis**: Track how patterns evolve through layers to understand processing dynamics

## Citation

If you use this code or methodology, please cite:

```bibtex
@misc{attention_geometry_2025,
  author = {Danan, Hillary},
  title = {Cross-Linguistic Attention Geometry in Transformer Models},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HillaryDanan/spanish-english-attention-geometry}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions or collaboration inquiries, please open an issue on this repository.

---

*This research prioritizes scientific rigor and reproducibility. All claims are grounded in peer-reviewed literature, with clear delineation between established findings and exploratory hypotheses.*
