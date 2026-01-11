# tasks/task3_temporal.py
# Task 3: Temporal Phase Prediction (Single-label)
#
# Usage (from repo root):
#   py tasks/task3_temporal.py
#
# Optional:
#   py tasks/task3_temporal.py --model mental/mental-bert-base-uncased

import os
import json
import random
import argparse
from typing import Dict, Any, List

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
PHASES = ["SEPARATION", "ANTICIPATION", "REUNION"]
PHASE2ID = {p: i for i, p in enumerate(PHASES)}
ID2PHASE = {i: p for p, i in PHASE2ID.items()}

LABEL_KEY = "phase_norm"   # normalized phase
IGNORE_LABEL = "UNKNOWN"   # excluded from training

# =====================================================
# UTILS
# =====================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((y_true == c) & (y_pred == c)).sum()
        fp = ((y_true != c) & (y_pred == c)).sum()
        fn = ((y_true == c) & (y_pred != c)).sum()
        if tp + fp + fn == 0:
            f1s.append(0.0)
        else:
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append((2 * p * r) / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s))


# =====================================================
# DATASET
# =====================================================
class RSPCPhaseDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        self.items = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                phase = ex.get(LABEL_KEY)
                if phase == IGNORE_LABEL or phase not in PHASE2ID:
                    continue  # skip UNKNOWN

                self.items.append({
                    "id": ex.get("id", ""),
                    "text": (ex.get("text") or "").strip(),
                    "label": PHASE2ID[phase],
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate_fn(batch: List[Dict[str, Any]], tokenizer, max_length: int):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"], labels


# =====================================================
# MODEL
# =====================================================
class PhaseClassifier(nn.Module):
    def __init__(self, backbone: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


# =====================================================
# EVAL
# =====================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)

        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = (y_true == y_pred).mean()
    mf1 = macro_f1(y_true, y_pred, len(PHASES))
    return acc, mf1


# =====================================================
# MAIN
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/splits/train.jsonl")
    parser.add_argument("--val", default="data/splits/val.jsonl")
    parser.add_argument("--test", default="data/splits/test.jsonl")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("runs/task3_temporal", exist_ok=True)
    best_path = "runs/task3_temporal/best.pt"

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    train_ds = RSPCPhaseDataset(args.train, tokenizer, args.max_length)
    val_ds   = RSPCPhaseDataset(args.val, tokenizer, args.max_length)
    test_ds  = RSPCPhaseDataset(args.test, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
    )

    model = PhaseClassifier(args.model, num_labels=len(PHASES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(
            f"[Epoch {epoch:02d}] loss={total_loss/len(train_loader):.4f} | "
            f"val_acc={val_acc:.4f} | val_macroF1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Saved best checkpoint (val_macroF1={val_f1:.4f})")

    # ---- Test
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc, test_f1 = evaluate(model, test_loader, device)

    print("\n===== TEST RESULTS (Task 3: Temporal Phase) =====")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Macro-F1 : {test_f1:.4f}")

    with open("runs/task3_temporal/test_metrics.json", "w") as f:
        json.dump({
            "accuracy": test_acc,
            "macro_f1": test_f1,
            "num_classes": len(PHASES),
            "ignored_label": IGNORE_LABEL,
        }, f, indent=2)

    print("Saved: runs/task3_temporal/test_metrics.json")
    print("===============================================")


if __name__ == "__main__":
    main()
