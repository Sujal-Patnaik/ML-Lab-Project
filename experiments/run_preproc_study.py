# Preprocessing study: does a paper-2 step help a paper-1 forest on the dataset it suits?
#     python experiments/run_preproc_study.py --dataset page+blocks+classification --variant resample \
#         --models DRaF --out results/preproc_study/parts/page_resample_DRaF.csv
#     python experiments/run_preproc_study.py --dataset magic+gamma+telescope --variant pca \
#         --models DRaF --out results/preproc_study/parts/magic_pca_DRaF.csv
# Variants (all fitted on the training split only, inside SVFPipeline.fit):
#     baseline : StandardScaler
#     resample : StandardScaler -> SMOTE-ENN        (for imbalanced data)
#     pca      : StandardScaler -> PCA(0.95)        (for correlated continuous features)
# Accuracy alone hides what happens to small classes, so balanced accuracy (mean per-class
# recall), macro F1 and the worst per-class recall are stored as well. The split and seed are
# the same as in run_accuracy.py, so the baseline accuracies match results/accuracy_uci.csv.
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.configs import ALL_MODELS, make_model  # noqa: E402
from experiments.run_accuracy import split_dataset  # noqa: E402
from svf.data import load_uci  # noqa: E402
from svf.evaluation import accuracy, confusion_matrix  # noqa: E402
from svf.pipeline import SVFPipeline  # noqa: E402

FIELDS = ["dataset", "model", "variant", "accuracy", "balanced_accuracy", "macro_f1", "min_recall",
          "fit_seconds", "mean_nodes", "n_train", "n_train_fit", "n_test", "seed"]

VARIANTS = {"baseline": dict(pca=None, resample=False),
            "resample": dict(pca=None, resample=True),
            "pca":      dict(pca=0.95, resample=False)}


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    # Balanced accuracy, macro F1 and worst recall are computed from the confusion matrix (C, C):
    #   recall_c = M[c,c] / row sum,  precision_c = M[c,c] / column sum (0 if c is never predicted)
    index = {}
    for i, c in enumerate(classes):
        index[c] = i
    a = np.array([index[v] for v in y_true])
    b = np.array([index[v] for v in y_pred])
    C = len(classes)
    M = confusion_matrix(a, b, C)
    recalls, f1s = [], []
    for c in range(C):
        tp = M[c, c]
        row = M[c, :].sum()
        col = M[:, c].sum()
        if row == 0:
            continue                      # class absent from the test set
        r = tp / row
        p = tp / col if col > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        recalls.append(r)
        f1s.append(f)
    return {"balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(np.mean(f1s)),
            "min_recall": float(min(recalls))}


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description="Baseline vs SMOTE-ENN vs global PCA on one dataset.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-trees", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    for name in args.models:
        if name not in ALL_MODELS:
            raise KeyError(f"unknown model '{name}', valid names: {list(ALL_MODELS)}")

    ds = load_uci(args.dataset, args.root)
    X_train, X_test, y_train, y_test = split_dataset(ds, args.seed)
    classes = np.unique(np.concatenate([y_train, y_test]))
    folder = os.path.dirname(args.out)
    if folder:
        os.makedirs(folder, exist_ok=True)
    # one row per model is appended to the output CSV
    for name in args.models:
        pipe = SVFPipeline(make_model(name, args.n_trees, args.seed), scale=True,
                           random_state=args.seed, **VARIANTS[args.variant])
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        y_pred = pipe.predict(X_test)
        row = {"dataset": ds.name, "model": name, "variant": args.variant,
               "accuracy": accuracy(y_test, y_pred), **class_metrics(y_test, y_pred, classes),
               "fit_seconds": round(fit_seconds, 3), "mean_nodes": float(pipe.model_.tree_sizes().mean()),
               "n_train": len(y_train), "n_train_fit": pipe.n_train_resampled_, "n_test": len(y_test),
               "seed": args.seed}
        new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
        with open(args.out, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(f"done  {ds.name:28s} {name:12s} {args.variant:8s} acc={row['accuracy']:.4f} "
              f"bal={row['balanced_accuracy']:.4f} f1={row['macro_f1']:.4f} fit={fit_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
