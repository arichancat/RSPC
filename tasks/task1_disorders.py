#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST Task 1: Multi-Label Disorder Classification (MLDC)
Optimized for GPU & CPU
- Mixed precision
- Efficient batching
- No per-sample tokenization overhead
- Fast numpy → tensor handling
"""

import os, json, time, random, argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup
)

# -------------------------------
# Label inventory
# -------------------------------
LABELS = ["SAD", "ADJ", "GAD", "MDD", "INSOMNIA"]
LABEL2ID = {l:i for i,l in enumerate(LABELS)}

# -------------------------------
# Reproducibility
# -------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------------
# Dataset (FAST)
# -------------------------------
class RSPCDisorderDataset(Dataset):
    def __init__(self, path):
        self.rows = []
        with open(path,"r",encoding="utf8") as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        x = self.rows[i]
        text = x.get("text","")
        labs = x.get("disorder_norm",[])

        y = np.zeros(len(LABELS),dtype=np.float32)
        for l in labs:
            if l in LABEL2ID:
                y[LABEL2ID[l]] = 1.0

        return text, y

# -------------------------------
# Collator (FAST TOKENIZATION)
# -------------------------------
@dataclass
class Collate:
    tokenizer: Any
    max_length:int=256

    def __call__(self,batch):
        texts, labels = zip(*batch)
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return enc["input_ids"], enc["attention_mask"], torch.tensor(labels)

# -------------------------------
# Model
# -------------------------------
class MultiLabel(nn.Module):
    def __init__(self,name,num_labels):
        super().__init__()
        cfg = AutoConfig.from_pretrained(name)
        self.enc = AutoModel.from_pretrained(name,config=cfg)
        self.head = nn.Linear(cfg.hidden_size,num_labels)

    def forward(self,iids,mask):
        o = self.enc(input_ids=iids,attention_mask=mask)
        cls = o.last_hidden_state[:,0]
        return self.head(cls)

# -------------------------------
# Metrics
# -------------------------------
def sigmoid(x): return 1/(1+np.exp(-x))

def f1(tp,fp,fn):
    d = 2*tp+fp+fn
    return 0 if d==0 else (2*tp)/d

def compute_scores(y_true,y_prob,thr=0.5):
    y_pred=(y_prob>=thr).astype(int)
    macro=[]
    tp=fp=fn=0

    for j in range(len(LABELS)):
        t=y_true[:,j]; p=y_pred[:,j]
        tpj=((p==1)&(t==1)).sum()
        fpj=((p==1)&(t==0)).sum()
        fnj=((p==0)&(t==1)).sum()
        tp+=tpj; fp+=fpj; fn+=fnj
        macro.append(f1(tpj,fpj,fnj))

    return float(np.mean(macro)), f1(tp,fp,fn)

# -------------------------------
# EVAL LOOP (with AMP)
# -------------------------------
@torch.no_grad()
def evaluate(model,loader,device):
    model.eval()
    all_true=[]; all_logits=[]
    for ids,mask,lab in loader:
        ids,mask=ids.to(device),mask.to(device)
        logits=model(ids,mask).cpu().numpy()
        all_logits.append(logits)
        all_true.append(lab.numpy())

    y_true=np.concatenate(all_true)
    y_prob=sigmoid(np.concatenate(all_logits))

    macro,micro=compute_scores(y_true,y_prob)
    return macro,micro

# -------------------------------
# TRAIN LOOP (AMP & Optimized)
# -------------------------------
def train_epoch(model,loader,opt,sched,crit,device,scaler):
    model.train()
    losses=[]
    for ids,mask,lab in loader:
        ids,mask,lab=ids.to(device),mask.to(device),lab.to(device)

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits=model(ids,mask)
            loss=crit(logits,lab)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        sched.step()
        losses.append(loss.item())

    return float(np.mean(losses))

# -------------------------------
# MAIN
# -------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",default="data/splits/train.jsonl")
    ap.add_argument("--val",default="data/splits/val.jsonl")
    ap.add_argument("--test",default="data/splits/test.jsonl")
    ap.add_argument("--model",default="bert-base-uncased")
    ap.add_argument("--batch",type=int,default=16)
    ap.add_argument("--epochs",type=int,default=4)
    ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default="runs/task1_disorders_fast")
    args=ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out,exist_ok=True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok=AutoTokenizer.from_pretrained(args.model,use_fast=True)
    col=Collate(tok)

    train=DataLoader(RSPCDisorderDataset(args.train),batch_size=args.batch,
                     shuffle=True,collate_fn=col,num_workers=2)

    val=DataLoader(RSPCDisorderDataset(args.val),batch_size=args.batch,
                   shuffle=False,collate_fn=col,num_workers=2)

    test=DataLoader(RSPCDisorderDataset(args.test),batch_size=args.batch,
                    shuffle=False,collate_fn=col,num_workers=2)

    model=MultiLabel(args.model,len(LABELS)).to(device)

    crit=nn.BCEWithLogitsLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    steps=len(train)*args.epochs
    sched=get_linear_schedule_with_warmup(opt,int(steps*0.1),steps)

    scaler=torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best=-1
    for e in range(1,args.epochs+1):
        loss=train_epoch(model,train,opt,sched,crit,device,scaler)
        macro,micro=evaluate(model,val,device)
        print(f"[E{e}] loss={loss:.4f} val_macro={macro:.4f}")

        if macro>best:
            best=macro
            torch.save(model.state_dict(),os.path.join(args.out,"best.pt"))

    model.load_state_dict(torch.load(os.path.join(args.out,"best.pt"),map_location=device))
    macro,micro=evaluate(model,test,device)

    out={
        "macro_f1":macro,
        "micro_f1":micro
    }

    with open(os.path.join(args.out,"test_metrics.json"),"w") as f:
        json.dump(out,f,indent=2)

    print("\nFINAL TEST RESULTS")
    print(out)

if __name__=="__main__":
    main()
