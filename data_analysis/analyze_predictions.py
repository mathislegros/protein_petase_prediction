"""
Statistical and visual analysis of zero-shot PETase predictions.
Covers distributions, position effects, amino acid preferences, and epistasis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

OUTPUT_DIR = "outputs/"


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['num_mutations'] = df['mutations'].apply(lambda x: 0 if x == 'WT' else x.count(',') + 1)
    df['mutation_type'] = df['num_mutations'].apply(
        lambda x: 'WT' if x == 0 else 'Single' if x == 1 else 'Multi'
    )
    return df


def extract_single_mutant_info(df: pd.DataFrame) -> pd.DataFrame:
    single = df[df['mutation_type'] == 'Single'].copy()
    parsed = [re.match(r'([A-Z])(\d+)([A-Z])', m.strip()) for m in single['mutations']]
    single['position'] = [int(m.group(2)) if m else None for m in parsed]
    single['wt_aa']    = [m.group(1) if m else None for m in parsed]
    single['mut_aa']   = [m.group(3) if m else None for m in parsed]
    return single.dropna(subset=['position'])


def analyze_distributions(df: pd.DataFrame):
    print("Overall activity statistics:")
    print(df['predicted_activity'].describe().round(4))
    stat, p = stats.shapiro(df['predicted_activity'].sample(min(5000, len(df))))
    print(f"Shapiro-Wilk: W={stat:.4f}, p={p:.2e} → {'normal' if p > 0.05 else 'non-normal'}")

    print("\nBy mutation type:")
    for t in ['WT', 'Single', 'Multi']:
        sub = df[df['mutation_type'] == t]['predicted_activity']
        if len(sub):
            print(f"  {t:6s} n={len(sub):5d}  mean={sub.mean():.4f}  std={sub.std():.4f}")


def statistical_tests(df: pd.DataFrame):
    groups = {t: df[df['mutation_type'] == t]['predicted_activity'].values
              for t in ['WT', 'Single', 'Multi']}

    for a, b in [('WT', 'Single'), ('WT', 'Multi'), ('Single', 'Multi')]:
        if len(groups[a]) and len(groups[b]):
            t, p = stats.ttest_ind(groups[a], groups[b])
            d = (groups[a].mean() - groups[b].mean()) / np.sqrt(
                (groups[a].std()**2 + groups[b].std()**2) / 2
            )
            print(f"{a} vs {b}: t={t:.3f}, p={p:.2e}, Cohen's d={d:.3f} "
                  f"({'sig' if p < 0.05 else 'ns'})")


def analyze_positions(df: pd.DataFrame) -> pd.DataFrame:
    single = extract_single_mutant_info(df)
    pos_stats = single.groupby('position')['predicted_activity'].agg(['count', 'mean', 'std'])
    print("\nTop 10 positions by mean predicted activity:")
    print(pos_stats.nlargest(10, 'mean').round(4))
    print("\nTop 10 most sensitive positions (high variance):")
    print(pos_stats.nlargest(10, 'std')[['count', 'mean', 'std']].round(4))
    return single


def plot_all(df: pd.DataFrame, single: pd.DataFrame):
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(df['predicted_activity'], bins=50, edgecolor='k', alpha=0.7)
    axes[0, 0].axvline(df['predicted_activity'].mean(), color='red', ls='--', label='mean')
    axes[0, 0].set_title('Overall Distribution')
    axes[0, 0].legend()

    for t in ['WT', 'Single', 'Multi']:
        sub = df[df['mutation_type'] == t]
        if len(sub):
            axes[0, 1].hist(sub['predicted_activity'], bins=30, alpha=0.5, label=f'{t} (n={len(sub)})')
    axes[0, 1].set_title('By Mutation Type')
    axes[0, 1].legend()

    data_box   = [df[df['mutation_type'] == t]['predicted_activity'].values
                  for t in ['WT', 'Single', 'Multi'] if len(df[df['mutation_type'] == t])]
    labels_box = [t for t in ['WT', 'Single', 'Multi'] if len(df[df['mutation_type'] == t])]
    axes[1, 0].boxplot(data_box, labels=labels_box)
    axes[1, 0].set_title('Activity Distribution (box)')

    axes[1, 1].scatter(df['l2_distance'], df['predicted_activity'], alpha=0.3, s=8)
    axes[1, 1].set_xlabel('L2 distance from WT')
    axes[1, 1].set_ylabel('Predicted activity')
    axes[1, 1].set_title('Embedding Distance vs Activity')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}distributions.png', dpi=200)
    plt.close()

    if len(single):
        pos_mean = single.groupby('position')['predicted_activity'].mean().sort_index()
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(pos_mean.index, pos_mean.values, alpha=0.7)
        ax.axhline(0, color='red', ls='--', alpha=0.5)
        ax.set_title('Mean Activity by Sequence Position (single mutants)')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}position_effects.png', dpi=200)
        plt.close()

    metrics = [c for c in ['predicted_activity', 'cosine_similarity', 'l2_distance',
                            'embedding_change', 'three_di_similarity'] if c in df.columns]
    if len(metrics) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[metrics].corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, ax=ax)
        ax.set_title('Metric Correlations')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}metric_correlations.png', dpi=200)
        plt.close()

    print(f"Plots saved to {OUTPUT_DIR}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/predictions_structure.csv')
    args = parser.parse_args()

    df = load(args.input)
    print(f"Loaded {len(df)} sequences\n")

    analyze_distributions(df)
    print()
    statistical_tests(df)
    single = analyze_positions(df)
    plot_all(df, single)


if __name__ == "__main__":
    main()
