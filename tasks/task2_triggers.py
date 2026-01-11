# tasks/task2_triggers.py
# Multi-label Trigger Detection (Task 2) for RSPC
#
# Usage (run from repo root):
#   py tasks/task2_triggers.py
#
# Optional args:
#   py tasks/task2_triggers.py --model bert-base-uncased --epochs 6 --batch_size 8
#
# Outputs:
#   runs/task2_triggers/best.pt
#   runs/task2_triggers/test_metrics.json

import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

# =====================================================
# CONFIG
# =====================================================
LABEL_KEY = "triggers_norm"  # <-- normalized trigger list

CANON_TRIGGERS = [
    "COMMITMENT_AMBIGUITY",
    "LACK_COMMUNICATION",
    "REUNION_SEPARATION_STRESS",
    "TRUST_FIDELITY",
    "JEALOUSY_INSECURITY",
    "SILENCE_GAP",
    "SOCIAL_MEDIA_SURVEILLANCE",
    "TIMEZONE_MISMATCH",
]
LABEL2ID = {l: i for i, l in enumerate(CANON_TRIGGERS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


# =====================================================
# UTILS
# =====================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def f1_from_pr(p: float, r: float) -> float:
    return safe_div(2 * p * r, (p + r))


def compute_per_label_prf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    y_true/y_pred: (N, L) binary arrays
    Returns per-label precision/recall/f1 dicts.
    """
    per_p, per_r, per_f1 = {}, {}, {}
    for j in range(y_true.shape[1]):
        tp = int(((y_true[:, j] == 1) & (y_pred[:, j] == 1)).sum())
        fp = int(((y_true[:, j] == 0) & (y_pred[:, j] == 1)).sum())
        fn = int(((y_true[:, j] == 1) & (y_pred[:, j] == 0)).sum())
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        f1 = f1_from_pr(p, r)
        per_p[ID2LABEL[j]] = p
        per_r[ID2LABEL[j]] = r
        per_f1[ID2LABEL[j]] = f1
    return per_p, per_r, per_f1


def micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    return f1_from_pr(p, r)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, _, per_f1 = compute_per_label_prf(y_true, y_pred)
    return float(np.mean(list(per_f1.values()))) if per_f1 else 0.0


def roc_auc_per_label(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Simple AUROC per label with sklearn if available; fallback to NaN if not.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return {ID2LABEL[j]: float("nan") for j in range(y_true.shape[1])}

    out = {}
    for j in range(y_true.shape[1]):
        # AUROC undefined if only one class present
        if len(np.unique(y_true[:, j])) < 2:
            out[ID2LABEL[j]] = float("nan")
        else:
            out[ID2LABEL[j]] = float(roc_auc_score(y_true[:, j], y_score[:, j]))
    return out


# =====================================================
# DATASET
# =====================================================
class RSPCTriggerDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        self.items = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                text = (ex.get("text") or "").strip()
                labels = ex.get(LABEL_KEY, []) or []
                vec = np.zeros(len(CANON_TRIGGERS), dtype=np.float32)
                for lab in labels:
                    if lab in LABEL2ID:
                        vec[LABEL2ID[lab]] = 1.0

                self.items.append(
                    {"id": ex.get("id", ""), "text": text, "labels": vec}
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def collate_fn(batch: List[Dict[str, Any]], tokenizer, max_length: int) -> Batch:
    texts = [b["text"] for b in batch]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    # Fast: stack numpy arrays first
    labels_np = np.stack([b["labels"] for b in batch], axis=0)
    labels = torch.from_numpy(labels_np).float()
    return Batch(enc["input_ids"], enc["attention_mask"], labels)


# =====================================================
# MODEL
# =====================================================
class TriggerClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logits = self.classifier(cls)
        return logits


# =====================================================
# TRAIN / EVAL
# =====================================================
@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    all_true, all_score = [], []
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss()

    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attention = batch.attention_mask.to(device)
        labels = batch.labels.to(device)

        logits = model(input_ids, attention)
        loss = bce(logits, labels)
        total_loss += float(loss.item())

        scores = torch.sigmoid(logits).cpu().numpy()
        all_score.append(scores)
        all_true.append(labels.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_score = np.concatenate(all_score, axis=0)
    y_pred = (y_score >= threshold).astype(np.int32)

    metrics = {
        "loss": total_loss / max(1, len(loader)),
        "macro_f1": macro_f1(y_true, y_pred),
        "micro_f1": micro_f1(y_true, y_pred),
        "per_label_f1": compute_per_label_prf(y_true, y_pred)[2],
        "per_label_auroc": roc_auc_per_label(y_true, y_score),
    }

    # Macro AUROC ignoring NaNs
    aurocs = [v for v in metrics["per_label_auroc"].values() if not (isinstance(v, float) and math.isnan(v))]
    metrics["macro_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")

    return metrics


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip: float = 1.0):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attention = batch.attention_mask.to(device)
        labels = batch.labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attention)
        loss = bce(logits, labels)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


# =====================================================
# MAIN
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/splits/train.jsonl")
    parser.add_argument("--val", default="data/splits/val.jsonl")
    parser.add_argument("--test", default="data/splits/test.jsonl")
    parser.add_argument("--model", default="bert-base-uncased")  # change to mentalbert if you want
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("runs/task2_triggers", exist_ok=True)
    best_path = os.path.join("runs", "task2_triggers", "best.pt")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    train_ds = RSPCTriggerDataset(args.train, tokenizer, max_length=args.max_length)
    val_ds = RSPCTriggerDataset(args.val, tokenizer, max_length=args.max_length)
    test_ds = RSPCTriggerDataset(args.test, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
        num_workers=0,
    )

    model = TriggerClassifier(args.model, num_labels=len(CANON_TRIGGERS), dropout=0.1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_macro = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_loader, device, threshold=args.threshold)

        print(
            f"[Epoch {epoch:02d}] loss={train_loss:.4f} | "
            f"val_macroF1={val_metrics['macro_f1']:.4f} | "
            f"val_microF1={val_metrics['micro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro:
            best_val_macro = val_metrics["macro_f1"]
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, best_path)
            print(f"  ✅ Saved best checkpoint: {best_path} (val_macroF1={best_val_macro:.4f})")

    # ---- Test with best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(model, test_loader, device, threshold=args.threshold)

    print("\n===== TEST RESULTS (Task 2: Trigger Detection) =====")
    print(f"Macro-F1 : {test_metrics['macro_f1']:.4f}")
    print(f"Micro-F1 : {test_metrics['micro_f1']:.4f}")
    print("\nPer-label F1:")
    for k, v in test_metrics["per_label_f1"].items():
        print(f"  {k:<28}: {v:.4f}")
    print("\nPer-label AUROC:")
    for k, v in test_metrics["per_label_auroc"].items():
        print(f"  {k:<28}: {v}")
    print(f"Macro-AUROC: {test_metrics['macro_auroc']}")

    out_path = os.path.join("runs", "task2_triggers", "test_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nSaved: {out_path}")
    print("=====================================")


if __name__ == "__main__":
    main()
