"""
Baseline ESM-2 zero-shot predictor.
Uses cosine similarity, L2 distance, and normalized embedding change
between WT and mutant mean-pooled representations.
No training required.
"""

import torch
import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm
import esm
import torch.nn.functional as F


CSV_PATH    = "data/predictive-pet-zero-shot-test-2025.csv"
OUTPUT_PATH = "data/predictions_basic.csv"


class ESM2ZeroShotPredictor:
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D"):
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"ESM-2 loaded on {self.device}")

    def compute_scores(self, wt_sequence: str, mut_sequences: List[str], batch_size: int = 16) -> dict:
        wt_emb = self._encode([wt_sequence])

        cosine_list, l2_list, change_list = [], [], []
        for i in tqdm(range(0, len(mut_sequences), batch_size), desc="Scoring"):
            batch = mut_sequences[i:i + batch_size]
            embs  = self._encode(batch)
            wt_exp = wt_emb.expand(len(batch), -1)

            cosine = F.cosine_similarity(wt_exp, embs, dim=1)
            l2     = torch.norm(embs - wt_exp, p=2, dim=1)
            change = l2 / torch.norm(wt_emb, p=2)

            cosine_list.append(cosine.cpu().numpy())
            l2_list.append(l2.cpu().numpy())
            change_list.append(change.cpu().numpy())

        cos = np.concatenate(cosine_list)
        l2  = np.concatenate(l2_list)
        chg = np.concatenate(change_list)

        # Z-score each metric; flip cosine and flip l2/change sign
        # convention: higher score = more active
        def zscore(x): return (x - x.mean()) / (x.std() + 1e-8)
        ensemble = (-zscore(cos) + zscore(l2) + zscore(chg)) / 3.0

        return {
            'cosine_similarity': cos,
            'l2_distance': l2,
            'embedding_change': chg,
            'ensemble_score': ensemble,
            'predicted_activity': -ensemble,  # less disruption = more active
        }

    def _encode(self, sequences: List[str]) -> torch.Tensor:
        data = [(f"s{i}", s) for i, s in enumerate(sequences)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            out = self.model(tokens, repr_layers=[33], return_contacts=False)
            return out["representations"][33][:, 1:-1, :].mean(dim=1)


def detect_mutations(wt: str, mut: str) -> str:
    muts = [f"{w}{i+1}{m}" for i, (w, m) in enumerate(zip(wt, mut)) if w != m]
    return ', '.join(muts) if muts else 'WT'


def main():
    df = pd.read_csv(CSV_PATH)
    sequences   = df['sequence'].tolist()
    wt_sequence = sequences[0]

    df['mutations'] = [detect_mutations(wt_sequence, s) for s in tqdm(sequences, desc="Mutations")]

    predictor = ESM2ZeroShotPredictor()
    scores = predictor.compute_scores(wt_sequence, sequences, batch_size=16)
    for k, v in scores.items():
        df[k] = v

    df[['sequence', 'mutations', 'predicted_activity',
        'cosine_similarity', 'l2_distance', 'embedding_change', 'ensemble_score']].to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
