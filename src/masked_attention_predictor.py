"""
Masked Attention Zero-Shot Fitness Predictor.

The fitness label is appended as an extra token to the protein embedding sequence.
A transformer is trained with a masked prediction objective:
  - with probability p_label: mask the fitness token → train to predict fitness
  - with probability p_protein: mask a fraction of protein positions → train to reconstruct embeddings

At inference: mask only the fitness token. The model predicts fitness from the
unmasked protein context, generalizing to proteins not seen during training (zero-shot).

This formulation is similar to masked language modelling (BERT) applied to protein fitness,
where the fitness score is treated as a learnable "token" in the same attention space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import esm


@dataclass
class MaskedPredictorConfig:
    esm_dim: int = 1280          # ESM-2 650M hidden dim
    hidden_dim: int = 512
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    # Masking probabilities during training
    p_mask_label: float = 0.5      # probability of masking the fitness token
    p_mask_protein: float = 0.15   # fraction of protein positions masked when p_mask_label not triggered
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01


class MaskedFitnessPredictor(nn.Module):
    """
    Transformer operating over [protein_tokens... | fitness_token].

    Protein positions: ESM-2 embeddings projected to hidden_dim.
    Fitness token:     scalar label embedded to hidden_dim via a learned linear map.
    Mask token:        shared learned embedding replacing any masked position.

    During training, the model is jointly trained to:
      (1) reconstruct masked fitness labels (MSE loss)
      (2) reconstruct masked protein positions (cosine/MSE loss in ESM space)

    At inference, only the fitness token is masked → the output at that position
    is passed through the fitness head to produce the predicted score.
    """

    def __init__(self, config: MaskedPredictorConfig):
        super().__init__()
        self.config = config

        self.protein_proj = nn.Linear(config.esm_dim, config.hidden_dim)
        # Label is a scalar; embed it as a single vector
        self.label_proj   = nn.Linear(1, config.hidden_dim)
        # Shared mask token, learned
        self.mask_token   = nn.Parameter(torch.zeros(config.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim, nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Fitness prediction head (applied to the last token — the label position)
        self.fitness_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1),
        )
        # Protein reconstruction head (project back to ESM space for auxiliary loss)
        self.protein_reconstruct = nn.Linear(config.hidden_dim, config.esm_dim)

        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        protein_emb: torch.Tensor,      # (B, L, esm_dim)  or (B, esm_dim) if mean-pooled
        label: Optional[torch.Tensor],  # (B,) fitness scores; None = always mask
        mask_label: bool = True,
        protein_mask_idx: Optional[torch.Tensor] = None,  # (B, n_masked) indices to mask
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            fitness_pred  — (B, 1) predicted fitness at the label token position
            protein_recon — (B, n_masked, esm_dim) reconstructed masked protein embeddings,
                            or None if no protein positions were masked
        """
        B = protein_emb.shape[0]

        if protein_emb.dim() == 2:
            # Mean-pooled: treat as a single protein token
            protein_emb = protein_emb.unsqueeze(1)

        x_protein = self.protein_proj(protein_emb)  # (B, L, hidden_dim)

        # Optionally mask protein positions with the shared mask token
        if protein_mask_idx is not None:
            for b in range(B):
                x_protein[b, protein_mask_idx[b]] = self.mask_token

        # Build label token
        if label is not None and not mask_label:
            x_label = self.label_proj(label.view(B, 1, 1).float())  # (B, 1, hidden_dim)
        else:
            x_label = self.mask_token.view(1, 1, -1).expand(B, 1, -1)

        # Concatenate: [protein_0, ..., protein_L-1, fitness_label]
        x = torch.cat([x_protein, x_label], dim=1)  # (B, L+1, hidden_dim)
        out = self.transformer(x)

        # Fitness prediction from the last position (fitness token)
        fitness_pred = self.fitness_head(out[:, -1, :])  # (B, 1)

        # Protein reconstruction at masked positions (auxiliary training signal)
        protein_recon = None
        if protein_mask_idx is not None:
            recon_list = []
            for b in range(B):
                recon_list.append(self.protein_reconstruct(out[b, protein_mask_idx[b]]))
            protein_recon = torch.stack(recon_list)  # (B, n_masked, esm_dim)

        return fitness_pred, protein_recon


def training_step(
    model: MaskedFitnessPredictor,
    protein_emb: torch.Tensor,
    label: torch.Tensor,
    config: MaskedPredictorConfig,
    device: torch.device,
) -> dict:
    """
    Single training step with mixed masking strategy:
      - 50% of the time: mask the fitness label only
      - 50% of the time: mask protein positions only (and expose the label)
    Loss is MSE on whichever token(s) were masked.
    """
    B, L, _ = protein_emb.shape
    protein_emb = protein_emb.to(device)
    label = label.to(device).float()

    do_mask_label = torch.rand(1).item() < config.p_mask_label

    protein_mask_idx = None
    if not do_mask_label:
        # Mask a random fraction of protein positions
        n_mask = max(1, int(L * config.p_mask_protein))
        protein_mask_idx = torch.stack(
            [torch.randperm(L, device=device)[:n_mask] for _ in range(B)]
        )

    fitness_pred, protein_recon = model(
        protein_emb,
        label=label,
        mask_label=do_mask_label,
        protein_mask_idx=protein_mask_idx,
    )

    losses = {}
    if do_mask_label:
        # Reconstruct fitness label
        losses['label_loss'] = F.mse_loss(fitness_pred.squeeze(-1), label)
        total = losses['label_loss']
    else:
        # Reconstruct masked protein positions in ESM embedding space
        target = torch.stack([protein_emb[b, protein_mask_idx[b]] for b in range(B)])
        losses['protein_loss'] = F.mse_loss(protein_recon, target)
        total = losses['protein_loss']

    losses['total'] = total
    return losses


# ─── Inference ──────────────────────────────────────────────────────────────

class ZeroShotPipeline:
    """
    End-to-end pipeline: ESM-2 embedding + masked attention fitness prediction.
    Trained on labeled data from ProteinGym or similar; applied zero-shot to PETase.
    """

    def __init__(self, model_path: Optional[str] = None, config: Optional[MaskedPredictorConfig] = None):
        self.config = config or MaskedPredictorConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading ESM-2 (650M)...")
        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
        self.esm_model.eval().to(self.device)

        self.predictor = MaskedFitnessPredictor(self.config).to(self.device)
        if model_path:
            ckpt = torch.load(model_path, map_location=self.device)
            self.predictor.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded checkpoint: {model_path}")
        self.predictor.eval()

    @torch.no_grad()
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """ESM-2 mean-pooled embedding → (1, esm_dim)."""
        _, _, tokens = self.esm_batch_converter([("p", sequence)])
        tokens = tokens.to(self.device)
        out = self.esm_model(tokens, repr_layers=[33], return_contacts=False)
        return out["representations"][33][:, 1:-1, :].mean(dim=1)

    @torch.no_grad()
    def predict(self, sequences: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Predict fitness for each sequence.
        Label token is always masked — the model infers fitness from protein context alone.
        """
        scores = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Predicting"):
            batch = sequences[i:i + batch_size]
            embs  = torch.cat([self.encode_sequence(s) for s in batch], dim=0)  # (B, esm_dim)
            embs  = embs.unsqueeze(1)  # (B, 1, esm_dim) — single token per sequence

            pred, _ = self.predictor(embs, label=None, mask_label=True)
            scores.append(pred.squeeze(-1).cpu().numpy())

        return np.concatenate(scores)


# ─── Example: train from scratch on labeled data ─────────────────────────────

def train(
    model: MaskedFitnessPredictor,
    protein_embeddings: torch.Tensor,  # (N, L, esm_dim)
    labels: torch.Tensor,              # (N,)
    config: MaskedPredictorConfig,
    n_epochs: int = 50,
    batch_size: int = 32,
    save_path: str = "checkpoints/masked_predictor.pt",
):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    N = len(labels)
    best_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(N)
        epoch_losses = []

        for start in range(0, N, batch_size):
            batch_idx = idx[start:start + batch_size]
            emb   = protein_embeddings[batch_idx]
            lbl   = labels[batch_idx]
            loss_dict = training_step(model, emb, lbl, config, device)

            optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss_dict['total'].item())

        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch:3d}  loss={mean_loss:.4f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({'model_state_dict': model.state_dict(), 'config': config}, save_path)

    print(f"Training complete. Best loss: {best_loss:.4f}. Saved to {save_path}")


# ─── Main (inference only) ───────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',      default='data/predictive-pet-zero-shot-test-2025.csv')
    parser.add_argument('--output',     default='data/predictions_masked_attention.csv')
    parser.add_argument('--checkpoint', default=None, help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    sequences = df['sequence'].tolist()

    pipeline = ZeroShotPipeline(model_path=args.checkpoint)
    scores = pipeline.predict(sequences, batch_size=args.batch_size)

    df['predicted_activity'] = scores
    df[['sequence', 'predicted_activity']].to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
