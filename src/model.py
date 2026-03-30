"""
Ensemble model for zero-shot mutation effect prediction.
Combines ESM-2 sequence embeddings with optional 3Di structural tokens.
Inspired by: Gitter et al. (ensemble + structure → best ProteinGym results).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ZeroShotConfig:
    hidden_dim: int = 1280        # ESM-2 650M embedding dim
    structure_dim: int = 512
    ensemble_size: int = 5
    dropout: float = 0.1
    use_multi_label: bool = True
    num_functional_terms: int = 100
    use_structure: bool = True
    structure_type: str = "3di"   # "3di" or "coords"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01


class StructureEncoder(nn.Module):
    """Encodes 3Di token sequences or raw backbone coordinates."""

    def __init__(self, config: ZeroShotConfig):
        super().__init__()
        self.config = config

        if config.structure_type == "3di":
            # 3Di alphabet: 20 tokens from Foldseek's structural alphabet
            self.structure_embed = nn.Embedding(20, config.structure_dim)
            self.structure_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.structure_dim, nhead=8,
                    dim_feedforward=2048, dropout=config.dropout, batch_first=True
                ),
                num_layers=3
            )
        else:
            # Raw coords: N, CA, C, O → flatten to 12 dims per residue
            self.coord_encoder = nn.Sequential(
                nn.Linear(4 * 3, config.structure_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.structure_dim, config.structure_dim),
            )

    def forward(self, structure_input: torch.Tensor) -> torch.Tensor:
        if self.config.structure_type == "3di":
            return self.structure_encoder(self.structure_embed(structure_input)).mean(dim=1)
        else:
            B, L, _, _ = structure_input.shape
            return self.coord_encoder(structure_input.reshape(B, L, -1)).mean(dim=1)


class MutationEffectPredictor(nn.Module):
    """Single ensemble member: fuses WT/mutant embeddings + optional structure."""

    def __init__(self, config: ZeroShotConfig):
        super().__init__()
        self.config = config
        combined_dim = config.hidden_dim + (config.structure_dim if config.use_structure else 0)

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.mutation_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
        )
        self.effect_head = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1),
        )
        if config.use_multi_label:
            self.functional_head = nn.Sequential(
                nn.Linear(config.hidden_dim // 2, 256),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(256, config.num_functional_terms),
            )

    def forward(self, wt_emb, mut_emb, structure_emb=None) -> Dict[str, torch.Tensor]:
        if structure_emb is not None and self.config.use_structure:
            wt_feat  = torch.cat([wt_emb,  structure_emb], dim=-1)
            mut_feat = torch.cat([mut_emb, structure_emb], dim=-1)
        else:
            wt_feat, mut_feat = wt_emb, mut_emb

        wt_fused  = self.fusion(wt_feat)
        mut_fused = self.fusion(mut_feat)
        encoded   = self.mutation_encoder(torch.cat([wt_fused, mut_fused], dim=-1))
        out = {"effect_score": self.effect_head(encoded).squeeze(-1)}
        if self.config.use_multi_label:
            out["functional_terms"] = self.functional_head(encoded)
        return out


class ZeroShotEnsemble(nn.Module):
    """
    Ensemble of MutationEffectPredictor models with a shared structure encoder
    and learned per-member weights (softmax-normalized).
    """

    def __init__(self, config: ZeroShotConfig):
        super().__init__()
        self.config = config
        if config.use_structure:
            self.structure_encoder = StructureEncoder(config)
        self.predictors = nn.ModuleList(
            [MutationEffectPredictor(config) for _ in range(config.ensemble_size)]
        )
        self.ensemble_weights = nn.Parameter(
            torch.ones(config.ensemble_size) / config.ensemble_size
        )

    def forward(self, wt_emb, mut_emb, structure_input=None, return_individual=False):
        structure_emb = None
        if structure_input is not None and self.config.use_structure:
            structure_emb = self.structure_encoder(structure_input)

        preds = [p(wt_emb, mut_emb, structure_emb) for p in self.predictors]
        weights = F.softmax(self.ensemble_weights, dim=0)
        scores  = torch.stack([p["effect_score"] for p in preds], dim=0)
        out = {"effect_score": (scores * weights.view(-1, 1)).sum(dim=0)}

        if self.config.use_multi_label:
            logits = torch.stack([p["functional_terms"] for p in preds], dim=0)
            out["functional_terms"] = (logits * weights.view(-1, 1, 1)).sum(dim=0)

        if return_individual:
            out["individual_predictions"] = preds
            out["ensemble_weights"] = weights

        return out

    @staticmethod
    def _apply_mutation(sequence: str, mutation: str) -> str:
        wt_aa, pos, mut_aa = mutation[0], int(mutation[1:-1]) - 1, mutation[-1]
        assert sequence[pos] == wt_aa, f"WT mismatch at position {pos+1}"
        seq = list(sequence)
        seq[pos] = mut_aa
        return ''.join(seq)


class ZeroShotLoss(nn.Module):
    """MSE on effect score + 0.5 × BCE on functional terms (if enabled)."""

    def __init__(self, config: ZeroShotConfig):
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        loss = self.mse(predictions["effect_score"], targets["effect_score"])
        info = {"effect_loss": loss.item()}
        if self.config.use_multi_label and "functional_terms" in predictions:
            fl = self.bce(predictions["functional_terms"], targets["functional_terms"])
            loss = loss + 0.5 * fl
            info["functional_loss"] = fl.item()
        info["total_loss"] = loss.item()
        return loss, info


def create_zero_shot_model(config: Optional[ZeroShotConfig] = None) -> ZeroShotEnsemble:
    return ZeroShotEnsemble(config or ZeroShotConfig())


if __name__ == "__main__":
    config = ZeroShotConfig(ensemble_size=5, use_structure=True)
    model  = create_zero_shot_model(config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    wt_emb  = torch.randn(4, config.hidden_dim)
    mut_emb = torch.randn(4, config.hidden_dim)
    struct  = torch.randint(0, 20, (4, 100))
    out = model(wt_emb, mut_emb, struct)
    print(f"Effect score shape: {out['effect_score'].shape}")
