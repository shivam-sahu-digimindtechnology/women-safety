
"""
Physical Violence Detection Head Training Script
================================================
Reproducible PyTorch training script for fine-tuning a lightweight temporal classification head
on sequence features (e.g. CLIP ViT-B/32 or ResNet embeddings / Pose dynamics) for datasets
such as RWF-2000, Hockey Fights, or CCTV-Fights.

Usage:
    python training/train_violence_head.py --epochs 20 --batch-size 32 --lr 1e-3 --synthetic
"""

import os
import sys
import argparse
import time
from typing import Tuple, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TemporalViolenceHead(nn.Module):
    """
    Lightweight Temporal MLP/GRU Head for Video Action Classification.
    Accepts pooled visual feature sequences of shape (Batch, SeqLen, FeatDim).
    """
    def __init__(self, feat_dim: int = 512, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, feat_dim)
        gru_out, _ = self.gru(x)
        # Global temporal average pooling
        pooled = torch.mean(gru_out, dim=1)
        out = self.classifier(pooled)
        return out


class SyntheticSequenceDataset(Dataset):
    """Generates synthetic sequence feature batches for diagnostics, baseline unit tests, and validation."""
    def __init__(self, num_samples: int = 200, seq_len: int = 16, feat_dim: int = 512):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        
        # Class 0: Normal activity (lower variance, random walk around 0)
        # Class 1: Violent activity (high variance, sporadic spike perturbations)
        np.random.seed(42)
        self.labels = np.random.randint(0, 2, size=num_samples)
        self.features = np.zeros((num_samples, seq_len, feat_dim), dtype=np.float32)

        for i in range(num_samples):
            if self.labels[i] == 0:
                self.features[i] = np.random.randn(seq_len, feat_dim) * 0.1
            else:
                self.features[i] = np.random.randn(seq_len, feat_dim) * 0.3 + 0.2

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Computes binary classification metrics: Accuracy, Precision, Recall, F1."""
    acc = np.mean(preds == targets)
    tp = np.sum((preds == 1) & (targets == 1))
    fp = np.sum((preds == 1) & (targets == 0))
    fn = np.sum((preds == 0) & (targets == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for feats, labels in loader:
        feats = feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
    return epoch_loss, metrics


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device)
            labels = labels.to(device)

            outputs = model(feats)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
    return epoch_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train lightweight temporal violence classifier")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--feat-dim", type=int, default=512, help="Feature dimension (e.g. CLIP 512)")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length in frames")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model weights")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic dataset for training test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training Pipeline] Device: {device} | FeatDim: {args.feat_dim} | SeqLen: {args.seq_len}")

    # Dataset setup
    train_dataset = SyntheticSequenceDataset(num_samples=160, seq_len=args.seq_len, feat_dim=args.feat_dim)
    val_dataset = SyntheticSequenceDataset(num_samples=40, seq_len=args.seq_len, feat_dim=args.feat_dim)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = TemporalViolenceHead(feat_dim=args.feat_dim, hidden_dim=64, num_classes=2, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_f1 = 0.0
    out_model_path = os.path.join(args.output_dir, "violence_head.pt")

    print("\n--- Starting Training Loop ---")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{dt:.1f}s] "
            f"Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.3f} F1: {val_metrics['f1']:.3f}"
        )

        if val_metrics["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "feat_dim": args.feat_dim,
                "seq_len": args.seq_len
            }, out_model_path)

    print(f"\n[Training Complete] Best Val F1: {best_val_f1:.4f} -> Model saved to {out_model_path}\n")


if __name__ == "__main__":
    main()
