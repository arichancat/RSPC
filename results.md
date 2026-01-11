# 📊 RSPC — Experiment Results

---

## 📁 Dataset Overview

| Metric | Value |
|------|-----|
| Total Samples | 1799 |
| Multi-label Samples | 1331 (73.99%) |

---

## 🧠 Disorder Distribution

| Disorder | Count |
|--------|-----|
| ADJ | 1341 |
| GAD | 1279 |
| SAD | 783 |
| MDD | 306 |
| INSOMNIA | 21 |

---

## 🔗 Top Co-Morbidities

| Pair | Count |
|----------------|------|
| ADJ + GAD | 1040 |
| ADJ + SAD | 696 |
| GAD + SAD | 670 |
| ADJ + MDD | 215 |
| GAD + MDD | 188 |
| MDD + SAD | 87 |
| GAD + INSOMNIA | 15 |
| ADJ + INSOMNIA | 14 |
| INSOMNIA + SAD | 12 |
| INSOMNIA + MDD | 9 |

---

## 🎯 Trigger Distribution

| Trigger | Count |
|---------------------------|------|
| COMMITMENT_AMBIGUITY | 1191 |
| LACK_COMMUNICATION | 1110 |
| REUNION_SEPARATION_STRESS | 389 |
| TRUST_FIDELITY | 329 |
| JEALOUSY_INSECURITY | 324 |
| SILENCE_GAP | 102 |
| SOCIAL_MEDIA_SURVEILLANCE | 37 |
| TIMEZONE_MISMATCH | 29 |

---

## ⏳ Phase Distribution

| Phase | Count |
|-----------|------|
| SEPARATION | 1172 |
| ANTICIPATION | 322 |
| REUNION | 234 |
| UNKNOWN | 71 |

---

## 🧪 Dataset Splits

| Split | Samples |
|------|--------|
| Train | 1259 |
| Validation | 179 |
| Test | 361 |

---

# 📌 TF-IDF Baseline

| Task | Macro-F1 | Micro-F1 |
|----|----|----|
| Disorder (T1) | 0.3836 | 0.7211 |
| Trigger (T2) | 0.1934 | 0.6340 |

---

# 🧠 Task-1 — Disorder Classification (BERT-base)

| Metric | Value |
|------|------|
| Macro-F1 | 0.4518 |
| Micro-F1 | 0.7451 |
| Macro-AUROC | 0.5697 |

### Per-Label F1

| Label | F1 |
|----|----|
| ADJ | 0.8429 |
| GAD | 0.8260 |
| SAD | 0.5901 |
| MDD | 0.0000 |
| INSOMNIA | 0.0000 |

---

# 🎯 Task-2 — Trigger Detection (BERT-base)

| Metric | Value |
|----|----|
| Loss | 0.390 |
| Macro-F1 | 0.2943 |
| Micro-F1 | 0.6264 |
| Macro-AUROC | 0.7061 |

### Per-Label F1

| Trigger | F1 |
|----|----|
| COMMITMENT_AMBIGUITY | 0.7590 |
| LACK_COMMUNICATION | 0.7335 |
| REUNION_SEPARATION_STRESS | 0.2056 |
| TRUST_FIDELITY | 0.3617 |
| JEALOUSY_INSECURITY | 0.2947 |
| Others | 0.000 |

---

# ⏳ Task-3 — Temporal Phase Classification

| Metric | Value |
|------|------|
| Accuracy | 0.7040 |
| Macro-F1 | 0.5176 |

---

# 🔬 Model Comparison — Task-1

| Model | Macro-F1 | Micro-F1 |
|-----|-----|-----|
| TF-IDF | 0.3836 | 0.7211 |
| Bio-ClinicalBERT | 0.4253 | 0.7237 |
| BERT-base | 0.4397 | 0.7209 |
| **RoBERTa-base** | **0.4925** | **0.7299** |

---

# 🏆 Final Multi-Task System

| Model | T1 Macro-F1 | T1 Micro-F1 | T2 Macro-F1 | T2 Micro-F1 | T3 Acc | T3 Macro-F1 |
|-----|-----|-----|-----|-----|-----|-----|
| BERT-base | 0.4515 | 0.7451 | 0.2943 | 0.6264 | 0.7040 | 0.5176 |

---

## 📌 Key Observations

- Transformer models significantly outperform TF-IDF baseline  
- RoBERTa provides best disorder detection performance  
- Rare disorders (MDD, INSOMNIA) remain challenging due to data imbalance  
- Multi-task framework jointly models disorders, triggers, and temporal phase  
