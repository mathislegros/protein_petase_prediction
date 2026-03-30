"""
Biochemical plausibility check for top-ranked predictions.
Flags active-site mutations, charge reversals, and validates against
literature-known PETase beneficial mutations.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict

AA_GROUPS = {
    'hydrophobic': set('ACFGILMPVW'),
    'positive':    set('HKR'),
    'negative':    set('DE'),
    'aliphatic':   set('ILV'),
    'aromatic':    set('FHWY'),
}

# Literature-curated beneficial mutations (IsPETase, ThermoPETase, FAST-PETase)
KNOWN_BENEFICIAL = {
    'S121E': 'IsPETase — increased activity',
    'D186H': 'IsPETase — increased activity',
    'R224Q': 'IsPETase — increased activity',
    'N233K': 'IsPETase — increased activity',
    'S238F': 'ThermoPETase — thermostability',
    'W159H': 'ThermoPETase — thermostability',
    'S214H': 'FAST-PETase',
    'F238S': 'FAST-PETase',
    'D186S': 'FAST-PETase',
}

# Approximate catalytic triad region for IsPETase
ACTIVE_SITE = {160, 161, 162, 237, 238}


def parse_mutations(mut_str: str) -> List[Dict]:
    if mut_str == 'WT':
        return []
    muts = []
    for m in mut_str.split(', '):
        match = re.match(r'([A-Z])(\d+)([A-Z])', m.strip())
        if match:
            muts.append({'wt': match.group(1), 'pos': int(match.group(2)), 'mut': match.group(3), 'str': m.strip()})
    return muts


def substitution_type(wt: str, mut: str) -> str:
    conservative_groups = [('I', 'L', 'V'), ('F', 'Y', 'W'), ('K', 'R'), ('D', 'E'), ('S', 'T'), ('N', 'Q')]
    if any(wt in g and mut in g for g in conservative_groups):
        return 'conservative'
    wt_charge  = (1 if wt in 'KR' else -1 if wt in 'DE' else 0)
    mut_charge = (1 if mut in 'KR' else -1 if mut in 'DE' else 0)
    if abs(wt_charge - mut_charge) == 2:
        return 'charge_reversal'
    if wt_charge != mut_charge:
        return 'charge_change'
    if (wt in AA_GROUPS['hydrophobic']) != (mut in AA_GROUPS['hydrophobic']):
        return 'hydrophobicity_change'
    if 'G' in (wt, mut):
        return 'glycine_substitution'
    if 'P' in (wt, mut):
        return 'proline_substitution'
    return 'moderate'


def validate(df: pd.DataFrame, top_n: int = 100):
    top = df.nlargest(top_n, 'predicted_activity')
    all_muts = []
    for _, row in top.iterrows():
        for m in parse_mutations(row['mutations']):
            m['predicted_activity'] = row['predicted_activity']
            m['sub_type'] = substitution_type(m['wt'], m['mut'])
            m['in_active_site']   = m['pos'] in ACTIVE_SITE
            m['known_beneficial'] = m['str'] in KNOWN_BENEFICIAL
            all_muts.append(m)

    if not all_muts:
        print("No mutations to analyze.")
        return

    sub_counts = {}
    for m in all_muts:
        sub_counts[m['sub_type']] = sub_counts.get(m['sub_type'], 0) + 1
    total = sum(sub_counts.values())

    print(f"Substitution types in top {top_n}:")
    for t, c in sorted(sub_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:30s}: {c:4d} ({100*c/total:.1f}%)")

    known = [m for m in all_muts if m['known_beneficial']]
    print(f"\nKnown beneficial mutations found: {len(known)}")
    for m in known:
        print(f"  {m['str']:10s} → {KNOWN_BENEFICIAL[m['str']]}  (score={m['predicted_activity']:+.4f})")

    active = [m for m in all_muts if m['in_active_site']]
    if active:
        print(f"\nActive site mutations ({len(active)}) — high risk:")
        for m in active[:10]:
            print(f"  {m['str']:10s}  type={m['sub_type']}  score={m['predicted_activity']:+.4f}")
    else:
        print("\nNo active site mutations in top predictions (conservative).")

    n_conservative = sub_counts.get('conservative', 0)
    print(f"\nConservative: {n_conservative}/{total} ({100*n_conservative/total:.1f}%)")

    charge_rev = sub_counts.get('charge_reversal', 0)
    if charge_rev > total * 0.1:
        print(f"High charge-reversal rate ({charge_rev}) — consider filtering.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/predictions_structure.csv')
    parser.add_argument('--top_n', type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    validate(df, top_n=args.top_n)


if __name__ == "__main__":
    main()
