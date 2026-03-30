# Zero-Shot PETase Mutation Fitness Prediction

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
![ESM](https://img.shields.io/badge/ESM--2-650M-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

Zero-shot prediction of PETase enzyme mutation fitness using protein language models. Developed for the ALIGN competition (ETH Zürich, 2025). No mutation-specific training data required.

**Evaluation metric:** NDCG (Normalized Discounted Cumulative Gain) — ranking quality of beneficial mutations.

---

## Approach

### Final Method — Masked Attention Fitness Predictor

The fitness label is appended as a special token to the protein's ESM-2 embedding sequence. A transformer is trained with a masked prediction objective:

- **Label masking** (50%): mask the fitness token → model predicts fitness from protein context
- **Protein masking** (50%): mask random protein positions → model reconstructs ESM embeddings

At inference the fitness token is always masked; the model generalizes zero-shot to PETase mutations by conditioning on the protein embedding alone.

### Earlier Approaches (in `experiments/`)

| Method | Features |
|---|---|
| `predict_basic.py` | ESM-2 cosine similarity + L2 distance |
| `predict_improved.py` | + per-residue log-likelihoods + physicochemical deltas |
| `src/predict.py` | + simplified 3Di structural tokens (Foldseek-inspired), confidence-weighted |

---

## Repository Structure

```
├── src/
│   ├── masked_attention_predictor.py   # Final method: masked label + protein attention
│   ├── predict.py                      # ESM-2 + 3Di structure predictor
│   ├── model.py                        # ZeroShotEnsemble (sequence + structure)
│   ├── evaluate.py                     # NDCG and correlation metrics
│   └── utils.py                        # Dataset, PLM wrapper, trainer
├── experiments/
│   ├── predict_basic.py                # Baseline ESM-2 embedding similarity
│   └── predict_improved.py             # + log-likelihoods + physicochemistry
├── data_analysis/
│   ├── analyze_predictions.py          # Distribution, position, and epistasis analysis
│   └── biochemical_validation.py       # Conservative substitution and active-site checks
└── data/                               # Not tracked — place CSV files here
```

---

## Setup

```bash
pip install torch fair-esm pandas numpy scikit-learn tqdm scipy matplotlib seaborn
```

---

## Run

```bash
# Final method (requires trained checkpoint)
python src/masked_attention_predictor.py --input data/test.csv --checkpoint checkpoints/masked_predictor.pt

# Structure-aware method (no checkpoint needed)
python src/predict.py

# Analyze predictions
python data_analysis/analyze_predictions.py --input data/predictions_structure.csv

# Biochemical validation of top-100
python data_analysis/biochemical_validation.py --input data/predictions_structure.csv
```

---

## Notes

- Data files are not tracked. Place the competition CSV in `data/`.
- ESM-2 (650M) runs on CPU but is significantly faster with a GPU.
- The masked attention predictor requires pre-training on labeled protein fitness data (e.g., ProteinGym) before zero-shot transfer to PETase.
