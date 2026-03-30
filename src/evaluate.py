"""
NDCG-based evaluation for zero-shot mutation ranking.
NDCG is the official ALIGN tournament metric.
"""

import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import ndcg_score


def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int = None) -> float:
    """Compute NDCG@k. Shifts y_true to be non-negative (required by sklearn)."""
    y_true_pos = (y_true - y_true.min() + 1e-6).reshape(1, -1)
    return ndcg_score(y_true_pos, y_pred.reshape(1, -1), k=k)


def evaluate(preds: np.ndarray, targets: np.ndarray) -> dict:
    spearman = spearmanr(preds, targets)[0]
    pearson  = pearsonr(preds, targets)[0]
    mse  = np.mean((preds - targets) ** 2)
    mae  = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(mse)

    ndcg_full = compute_ndcg(targets, preds)
    ndcg_10   = compute_ndcg(targets, preds, k=10)
    ndcg_20   = compute_ndcg(targets, preds, k=20)
    ndcg_50   = compute_ndcg(targets, preds, k=50)
    ndcg_100  = compute_ndcg(targets, preds, k=100)

    print(f"Spearman ρ : {spearman:.4f}")
    print(f"Pearson  r : {pearson:.4f}")
    print(f"RMSE       : {rmse:.4f}")
    print(f"NDCG (full): {ndcg_full:.4f}")
    print(f"NDCG@10    : {ndcg_10:.4f}")
    print(f"NDCG@20    : {ndcg_20:.4f}")
    print(f"NDCG@50    : {ndcg_50:.4f}")
    print(f"NDCG@100   : {ndcg_100:.4f}")

    # Top-k overlap
    for k in [10, 20, 50, 100]:
        overlap = len(set(np.argsort(preds)[-k:]) & set(np.argsort(targets)[-k:]))
        print(f"Top-{k:3d} overlap: {overlap}/{k} ({100*overlap/k:.0f}%)")

    return {
        'spearman': spearman, 'pearson': pearson,
        'mse': mse, 'mae': mae, 'rmse': rmse,
        'ndcg_full': ndcg_full, 'ndcg_10': ndcg_10,
        'ndcg_20': ndcg_20, 'ndcg_50': ndcg_50, 'ndcg_100': ndcg_100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, help='CSV with predicted_activity column')
    parser.add_argument('--labels',      required=True, help='CSV with true activity/fitness column')
    parser.add_argument('--target_col',  default='activity')
    parser.add_argument('--output',      default=None)
    args = parser.parse_args()

    pred_df   = pd.read_csv(args.predictions)
    label_df  = pd.read_csv(args.labels)
    preds   = pred_df['predicted_activity'].values
    targets = label_df[args.target_col].values

    results = evaluate(preds, targets)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
