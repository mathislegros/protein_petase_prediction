"""
Improved ESM-2 zero-shot predictor.
Adds per-residue masked log-likelihoods and physicochemical property deltas
to the baseline embedding similarity scores.
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List
from tqdm import tqdm
import esm
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


CSV_PATH    = "data/predictive-pet-zero-shot-test-2025.csv"
OUTPUT_PATH = "data/predictions_improved.csv"

# Kyte-Doolittle hydrophobicity, MW (Da), charge at pH 7
AA_PROPERTIES = {
    'A': {'charge':  0,   'hydrophobicity':  1.8, 'size':  89},
    'C': {'charge':  0,   'hydrophobicity':  2.5, 'size': 121},
    'D': {'charge': -1,   'hydrophobicity': -3.5, 'size': 133},
    'E': {'charge': -1,   'hydrophobicity': -3.5, 'size': 147},
    'F': {'charge':  0,   'hydrophobicity':  2.8, 'size': 165},
    'G': {'charge':  0,   'hydrophobicity': -0.4, 'size':  75},
    'H': {'charge':  0.5, 'hydrophobicity': -3.2, 'size': 155},
    'I': {'charge':  0,   'hydrophobicity':  4.5, 'size': 131},
    'K': {'charge':  1,   'hydrophobicity': -3.9, 'size': 146},
    'L': {'charge':  0,   'hydrophobicity':  3.8, 'size': 131},
    'M': {'charge':  0,   'hydrophobicity':  1.9, 'size': 149},
    'N': {'charge':  0,   'hydrophobicity': -3.5, 'size': 132},
    'P': {'charge':  0,   'hydrophobicity': -1.6, 'size': 115},
    'Q': {'charge':  0,   'hydrophobicity': -3.5, 'size': 146},
    'R': {'charge':  1,   'hydrophobicity': -4.5, 'size': 174},
    'S': {'charge':  0,   'hydrophobicity': -0.8, 'size': 105},
    'T': {'charge':  0,   'hydrophobicity': -0.7, 'size': 119},
    'V': {'charge':  0,   'hydrophobicity':  4.2, 'size': 117},
    'W': {'charge':  0,   'hydrophobicity': -0.9, 'size': 204},
    'Y': {'charge':  0,   'hydrophobicity': -1.3, 'size': 181},
}


class ImprovedESM2Predictor:
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D"):
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self.model.to(self.device)
        print(f"ESM-2 loaded on {self.device}")

    def log_likelihood_score(self, wt_seq: str, mut_seq: str) -> float:
        """
        Masked marginal log-probability at mutated positions.
        More position-specific than embedding distance for single mutations.
        """
        mut_positions = [i for i, (w, m) in enumerate(zip(wt_seq, mut_seq)) if w != m]
        if not mut_positions:
            return 0.0

        data = [("protein", mut_seq)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            logits = self.model(tokens, repr_layers=[33])["logits"]
        log_probs = F.log_softmax(logits, dim=-1)

        score = sum(
            log_probs[0, pos + 1, self.alphabet.get_idx(mut_seq[pos])].item()
            for pos in mut_positions
        )
        return score / len(mut_positions)

    def physicochemical_deltas(self, wt_seq: str, mut_seq: str) -> Dict[str, float]:
        deltas = {'charge': 0.0, 'hydrophobicity': 0.0, 'size': 0.0, 'n_mutations': 0}
        for w, m in zip(wt_seq, mut_seq):
            if w != m and w in AA_PROPERTIES and m in AA_PROPERTIES:
                for prop in ('charge', 'hydrophobicity', 'size'):
                    deltas[prop] += abs(AA_PROPERTIES[m][prop] - AA_PROPERTIES[w][prop])
                deltas['n_mutations'] += 1
        if deltas['n_mutations'] > 0:
            for prop in ('charge', 'hydrophobicity', 'size'):
                deltas[prop] /= deltas['n_mutations']
        return deltas

    def embedding_scores(self, wt_seq: str, mut_seq: str) -> Dict[str, float]:
        wt_emb  = self._encode(wt_seq)
        mut_emb = self._encode(mut_seq)
        cosine  = F.cosine_similarity(wt_emb, mut_emb, dim=1).item()
        l2      = torch.norm(mut_emb - wt_emb, p=2).item()
        return {'cosine_similarity': cosine, 'l2_distance': l2,
                'embedding_change': l2 / torch.norm(wt_emb, p=2).item()}

    def predict_batch(self, wt_seq: str, sequences: List[str], batch_size: int = 8) -> pd.DataFrame:
        results = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Scoring"):
            for mut_seq in sequences[i:i + batch_size]:
                row = {}
                row['log_likelihood'] = self.log_likelihood_score(wt_seq, mut_seq)
                row.update(self.physicochemical_deltas(wt_seq, mut_seq))
                row.update(self.embedding_scores(wt_seq, mut_seq))
                results.append(row)
        return pd.DataFrame(results)

    def _encode(self, sequence: str) -> torch.Tensor:
        _, _, tokens = self.batch_converter([("p", sequence)])
        tokens = tokens.to(self.device)
        with torch.no_grad():
            out = self.model(tokens, repr_layers=[33])
            return out["representations"][33][:, 1:-1, :].mean(dim=1)


def create_ensemble_score(df: pd.DataFrame) -> np.ndarray:
    """
    Weighted sum over standardized features.
    log_likelihood and cosine are "good" (positive); the rest penalize disruption.
    """
    features = ['log_likelihood', 'charge', 'hydrophobicity', 'size',
                'cosine_similarity', 'l2_distance', 'embedding_change']
    X = StandardScaler().fit_transform(df[features].values)
    weights = [3.0, -1.5, -1.0, -0.5, 2.0, -2.0, -1.5]
    total_w = sum(abs(w) for w in weights)
    return sum(w * X[:, i] for i, w in enumerate(weights)) / total_w


def detect_mutations(wt: str, mut: str) -> str:
    muts = [f"{w}{i+1}{m}" for i, (w, m) in enumerate(zip(wt, mut)) if w != m]
    return ', '.join(muts) if muts else 'WT'


def main():
    df = pd.read_csv(CSV_PATH)
    sequences   = df['sequence'].tolist()
    wt_sequence = sequences[0]

    df['mutations'] = [detect_mutations(wt_sequence, s) for s in tqdm(sequences, desc="Mutations")]

    predictor = ImprovedESM2Predictor()
    scores_df = predictor.predict_batch(wt_sequence, sequences, batch_size=8)
    df = pd.concat([df, scores_df], axis=1)
    df['predicted_activity'] = create_ensemble_score(df)

    out_cols = ['sequence', 'mutations', 'predicted_activity', 'log_likelihood',
                'charge', 'hydrophobicity', 'size', 'n_mutations',
                'cosine_similarity', 'l2_distance', 'embedding_change']
    df[out_cols].to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
