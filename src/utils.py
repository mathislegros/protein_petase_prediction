"""
Shared utilities: dataset, PLM wrapper, trainers, metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    eval_every_n_steps: int = 100
    save_best_model: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")


class MutationDataset(Dataset):
    def __init__(self, mutations: List[Dict[str, Any]], wt_sequence: str, is_labeled: bool = True):
        self.mutations  = mutations
        self.wt_sequence = wt_sequence
        self.is_labeled = is_labeled

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, idx):
        data = self.mutations[idx]
        mutant_seq = self._apply_mutation(self.wt_sequence, data['mutation'])
        item = {
            'wt_sequence': self.wt_sequence,
            'mutant_sequence': mutant_seq,
            'mutation': data['mutation'],
            'is_labeled': self.is_labeled,
        }
        if self.is_labeled:
            item['effect'] = data['effect']
            if 'functional_terms' in data:
                item['functional_terms'] = data['functional_terms']
        return item

    @staticmethod
    def _apply_mutation(sequence: str, mutation: str) -> str:
        wt_aa, pos, mut_aa = mutation[0], int(mutation[1:-1]) - 1, mutation[-1]
        assert sequence[pos] == wt_aa, f"WT mismatch at position {pos+1}"
        seq = list(sequence)
        seq[pos] = mut_aa
        return ''.join(seq)


class ProteinLanguageModelWrapper:
    """ESM-2 wrapper with mean-pooled sequence embeddings."""

    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", device: str = "cuda"):
        self.device = device
        try:
            import esm
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            self.model = self.model.to(device).eval()
            self.batch_converter = self.alphabet.get_batch_converter()
        except ImportError:
            print("ESM not installed. Run: pip install fair-esm")
            self.model = None

    def encode(self, sequences: List[str]) -> torch.Tensor:
        if self.model is None:
            return torch.randn(len(sequences), 1280, device=self.device)
        data = [(f"seq_{i}", s) for i, s in enumerate(sequences)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            out = self.model(tokens, repr_layers=[33])
            emb = out["representations"][33][:, 1:-1, :].mean(dim=1)
        return emb


class EvaluationMetrics:
    @staticmethod
    def compute_all(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        return {
            'spearman': spearmanr(predictions, targets)[0],
            'pearson':  pearsonr(predictions, targets)[0],
            'mse': np.mean((predictions - targets) ** 2),
            'mae': np.mean(np.abs(predictions - targets)),
        }


class ZeroShotTrainer:
    def __init__(self, model: nn.Module, plm: ProteinLanguageModelWrapper, config: TrainingConfig):
        self.model  = model.to(config.device)
        self.plm    = plm
        self.config = config
        self.device = config.device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        from src.model import ZeroShotLoss
        self.criterion = ZeroShotLoss(model.config)
        self.best_spearman = -1.0
        self.global_step   = 0

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            wt_emb  = batch['wt_embedding'].to(self.device)
            mut_emb = batch['mut_embedding'].to(self.device)
            targets = {'effect_score': batch['effect'].to(self.device)}
            if 'functional_terms' in batch:
                targets['functional_terms'] = batch['functional_terms'].to(self.device)

            preds = self.model(wt_emb, mut_emb)
            loss, info = self.criterion(preds, targets)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            losses.append(info['total_loss'])
            self.global_step += 1

        return {'train_loss': np.mean(losses)}

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        preds, targets = [], []
        for batch in eval_loader:
            out = self.model(batch['wt_embedding'].to(self.device), batch['mut_embedding'].to(self.device))
            preds.append(out['effect_score'].cpu().numpy())
            targets.append(batch['effect'].numpy())
        return EvaluationMetrics.compute_all(np.concatenate(preds), np.concatenate(targets))

    def train(self, train_loader, eval_loader):
        history = {'train_loss': [], 'val_spearman': [], 'val_mse': []}
        for epoch in range(self.config.num_epochs):
            train_m = self.train_epoch(train_loader, epoch)
            eval_m  = self.evaluate(eval_loader)
            history['train_loss'].append(train_m['train_loss'])
            history['val_spearman'].append(eval_m['spearman'])
            history['val_mse'].append(eval_m['mse'])
            print(f"Epoch {epoch}: loss={train_m['train_loss']:.4f} spearman={eval_m['spearman']:.4f}")
            if eval_m['spearman'] > self.best_spearman:
                self.best_spearman = eval_m['spearman']
                if self.config.save_best_model:
                    self._save('best_model.pt')
        return history

    def _save(self, filename: str):
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_spearman':        self.best_spearman,
            'global_step':          self.global_step,
        }, self.config.save_dir / filename)
