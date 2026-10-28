# Accuracy experiment: every model on every dataset -> one CSV row per (dataset, model).
#     python experiments/run_accuracy.py --source toy --out results/accuracy_toy.csv
#     python experiments/run_accuracy.py --source uci --out results/accuracy_uci.csv
#     python experiments/run_accuracy.py --source toy --resample        (with SMOTE-ENN)
#     python experiments/run_accuracy.py --source toy --pca 0.95        (with global PCA)
# Rows already in the CSV are skipped and each new row is appended at once, so an
# interrupted run can be resumed. The same split and seed are used for every model on a
# dataset (paired comparison). The predefined split is used when the dataset has one,
# otherwise a stratified 70/30 split.
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import train_test_split  # noqa: E402

from experiments.configs import ALL_MODELS, PAPER_MODELS, make_model  # noqa: E402
from svf.data import TOY, Dataset, list_uci, load_toy, load_uci  # noqa: E402
from svf.evaluation import accuracy  # noqa: E402
from svf.pipeline import SVFPipeline  # noqa: E402

FIELDS = ["dataset", "model", "accuracy", "fit_seconds", "mean_nodes", "n_train", "n_test", "seed"]


def split_dataset(ds: Dataset, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # X_train, X_test, y_train, y_test: the predefined split if present, otherwise a stratified
    # 70/30 split (not stratified when some class has fewer than 2 samples).
    if ds.train_idx is not None and ds.test_idx is not None:
        return ds.X[ds.train_idx], ds.X[ds.test_idx], ds.y[ds.train_idx], ds.y[ds.test_idx]
    counts = {}
    for label in ds.y:
        counts[label] = counts.get(label, 0) + 1
    stratify = ds.y if min(counts.values()) >= 2 else None
    return train_test_split(ds.X, ds.y, test_size=0.3, stratify=stratify, random_state=seed)


def evaluate(model_name: str, ds: Dataset, seed: int, n_trees: int,
             resample: bool = False, pca=None) -> Dict[str, object]:
    # One model is fitted on one dataset (scaler, optional PCA / SMOTE-ENN, forest) and
    # its test accuracy, fit time and mean tree size are recorded.
    X_train, X_test, y_train, y_test = split_dataset(ds, seed)
    pipe = SVFPipeline(make_model(model_name, n_trees, seed), scale=True,
                       pca=pca, resample=resample, random_state=seed)
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    acc = accuracy(y_test, pipe.predict(X_test))
    return {"dataset": ds.name, "model": model_name, "accuracy": acc,
            "fit_seconds": round(fit_seconds, 3), "mean_nodes": float(pipe.model_.tree_sizes().mean()),
            "n_train": len(y_train), "n_test": len(y_test), "seed": seed}


def done_keys(path: str) -> Set[Tuple[str, str]]:
    # (dataset, model) pairs already present in the CSV (empty set if there is no file).
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            done.add((row["dataset"], row["model"]))
    return done


def append_row(path: str, row: Dict[str, object]) -> None:
    # One row is appended; the header is written first for a new file.
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({f: row[f] for f in FIELDS})


def iter_datasets(source: str, root: str, names: Iterable[str] = ()) -> Iterable[Dataset]:
    # Datasets are yielded one by one: the toy sets, or every UCI folder under root
    # (or only the given names).
    names = list(names)
    if source == "toy":
        for name in (names or TOY):
            yield load_toy(name)
    elif source == "uci":
        for name in (names or list_uci(root)):
            yield load_uci(name, root)
    else:
        raise ValueError("source must be 'toy' or 'uci'")


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description="Test accuracy of every model on every dataset.")
    ap.add_argument("--source", choices=["toy", "uci"], default="toy")
    ap.add_argument("--root", default="data")
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--models", nargs="*", default=list(PAPER_MODELS))
    ap.add_argument("--n-trees", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resample", action="store_true")
    ap.add_argument("--pca", default=None, help="int = components, float in (0,1) = explained variance")
    ap.add_argument("--out", default="results/accuracy.csv")
    args = ap.parse_args(argv)
    pca = None
    if args.pca is not None:
        pca = float(args.pca) if "." in args.pca else int(args.pca)
    for name in args.models:
        if name not in ALL_MODELS:
            raise KeyError(f"unknown model '{name}', valid names: {list(ALL_MODELS)}")

    done = done_keys(args.out)
    for ds in iter_datasets(args.source, args.root, args.datasets):
        for model_name in args.models:
            if (ds.name, model_name) in done:
                print(f"skip  {ds.name:28s} {model_name:12s} (already in {args.out})")
                continue
            row = evaluate(model_name, ds, args.seed, args.n_trees,
                           resample=args.resample, pca=pca)
            append_row(args.out, row)
            print(f"done  {ds.name:28s} {model_name:12s} acc={row['accuracy']:.4f} "
                  f"fit={row['fit_seconds']:.1f}s nodes={row['mean_nodes']:.0f}", flush=True)


if __name__ == "__main__":
    main()
