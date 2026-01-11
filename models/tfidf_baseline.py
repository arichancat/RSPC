# models/tfidf_baseline.py
# Baseline 0: TF-IDF + Logistic Regression
# Runs on Task 1 (Disorders) and Task 2 (Triggers)

import json
import argparse
import numpy as np
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score


# =====================================================
# LOAD DATA
# =====================================================
def load_jsonl(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            texts.append(ex["text"])
            labels.append(ex)
    return texts, labels


# =====================================================
# TASK 1 — DISORDER CLASSIFICATION
# =====================================================
def run_task1(train_path, test_path):
    print("\n===== TF-IDF BASELINE — TASK 1 (Disorders) =====")

    X_train, y_train_raw = load_jsonl(train_path)
    X_test,  y_test_raw  = load_jsonl(test_path)

    LABELS = ["ADJ", "GAD", "SAD", "MDD", "INSOMNIA"]
    label2id = {l: i for i, l in enumerate(LABELS)}

    def encode(y):
        vec = np.zeros(len(LABELS))
        for lab in y:
            if lab in label2id:
                vec[label2id[lab]] = 1
        return vec

    Y_train = np.array([encode(ex["disorder_norm"]) for ex in y_train_raw])
    Y_test  = np.array([encode(ex["disorder_norm"]) for ex in y_test_raw])

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=300, n_jobs=1)
    )
    clf.fit(X_train_vec, Y_train)

    Y_pred = clf.predict(X_test_vec)

    macro = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
    micro = f1_score(Y_test, Y_pred, average="micro", zero_division=0)

    print(f"Macro-F1 : {macro:.4f}")
    print(f"Micro-F1 : {micro:.4f}")

    return macro, micro


# =====================================================
# TASK 2 — TRIGGER DETECTION
# =====================================================
def run_task2(train_path, test_path):
    print("\n===== TF-IDF BASELINE — TASK 2 (Triggers) =====")

    X_train, y_train_raw = load_jsonl(train_path)
    X_test,  y_test_raw  = load_jsonl(test_path)

    LABELS = [
        "COMMITMENT_AMBIGUITY",
        "LACK_COMMUNICATION",
        "REUNION_SEPARATION_STRESS",
        "TRUST_FIDELITY",
        "JEALOUSY_INSECURITY",
        "SILENCE_GAP",
        "SOCIAL_MEDIA_SURVEILLANCE",
        "TIMEZONE_MISMATCH"
    ]
    label2id = {l: i for i, l in enumerate(LABELS)}

    def encode(y):
        vec = np.zeros(len(LABELS))
        for lab in y:
            if lab in label2id:
                vec[label2id[lab]] = 1
        return vec

    Y_train = np.array([encode(ex["triggers_norm"]) for ex in y_train_raw])
    Y_test  = np.array([encode(ex["triggers_norm"]) for ex in y_test_raw])

    vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=300, n_jobs=1)
    )
    clf.fit(X_train_vec, Y_train)

    Y_pred = clf.predict(X_test_vec)

    macro = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
    micro = f1_score(Y_test, Y_pred, average="micro", zero_division=0)

    print(f"Macro-F1 : {macro:.4f}")
    print(f"Micro-F1 : {micro:.4f}")

    return macro, micro


# =====================================================
# MAIN
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/splits/train.jsonl")
    parser.add_argument("--test",  default="data/splits/test.jsonl")
    args = parser.parse_args()

    run_task1(args.train, args.test)
    run_task2(args.train, args.test)

    print("\nTF-IDF baseline complete.")


if __name__ == "__main__":
    main()
