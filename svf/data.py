# Dataset loading.
#   - toy datasets from scikit-learn: iris, wine, breast_cancer, digits
#   - UCI datasets read from local files in data/<dataset_name>/
# Every loader returns a Dataset, so the experiment scripts do not depend on the source.
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# A dataset with an optional predefined split.
#   X : (N, n) float;  y : (N,) labels (raw, encoded later by the forest)
#   train_idx, test_idx : index arrays, or None when the caller makes its own split
@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    train_idx: Optional[np.ndarray] = None
    test_idx: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


TOY = ("iris", "wine", "breast_cancer", "digits")


def load_toy(name: str) -> Dataset:
    # A scikit-learn toy dataset is loaded by name (no predefined split).
    import sklearn.datasets
    if name not in TOY:
        raise ValueError(f"unknown toy dataset '{name}', choose from {TOY}")
    X, y = getattr(sklearn.datasets, "load_" + name)(return_X_y=True)
    return Dataset(name, np.asarray(X, dtype=np.float64), y)


def list_uci(root: str = "data") -> List[str]:
    # Sorted names of the dataset folders inside root ([] if root is missing).
    if not os.path.isdir(root):
        return []
    names = []
    for f in os.listdir(root):
        if os.path.isdir(os.path.join(root, f)):
            names.append(f)
    return sorted(names)


def load_uci(name: str, root: str = "data") -> Dataset:
    # One UCI dataset is read from root/name/. File layout of the raw UCI downloads:
    #   <name>.data             one file, no predefined split    (magic04, page-blocks)
    #   <name>.trn + <name>.tst predefined train / test files    (statlog shuttle)
    # Values are separated by commas or whitespace; the label is the LAST column (kept as a
    # string). Files compressed with Unix compress (.Z) are read through gzip -dc.
    folder = os.path.join(root, name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"no dataset folder {folder}")
    files = {}
    for f in sorted(os.listdir(folder)):
        if ":" in f:                     # skip Windows 'Zone.Identifier' streams
            continue
        base = f[:-2] if f.endswith(".Z") else f
        ext = os.path.splitext(base)[1]
        if ext in (".data", ".trn", ".tst"):
            if ext not in files or not f.endswith(".Z"):   # prefer the uncompressed copy
                files[ext] = os.path.join(folder, f)
    if ".trn" in files and ".tst" in files:
        # train rows first, then test rows; the split is stored as index arrays
        X_tr, y_tr = _read_table(files[".trn"])
        X_te, y_te = _read_table(files[".tst"])
        X = np.vstack([X_tr, X_te]); y = np.concatenate([y_tr, y_te])
        train_idx = np.arange(len(y_tr)); test_idx = np.arange(len(y_tr), len(y))
    elif ".data" in files:
        X, y = _read_table(files[".data"])
        train_idx, test_idx = None, None
    else:
        raise ValueError(f"{folder}: expected a .data file or a .trn/.tst pair")
    # sanity checks on a predefined split: disjoint, in range, every class present in train
    if train_idx is not None:
        N = len(y)
        if(len(np.intersect1d(train_idx, test_idx))>0):
            raise ValueError(f"{name}: train and test indices overlap")
        if(train_idx.min()<0 or test_idx.max()>=N):
            raise ValueError(f"{name}: index outside 0..{N-1}")
        missing = [c for c in np.unique(y) if c not in set(y[train_idx])]
        if missing:
            raise ValueError(f"{name}: classes {missing} never appear in the training set")
    return Dataset(name, X, y, train_idx, test_idx, meta={"files": files})


def _read_table(path: str):
    # One raw table is read -> X float64 (N, n), y str (N,).
    if path.endswith(".Z"):
        text = subprocess.run(["gzip", "-dc", path], check=True, capture_output=True).stdout.decode()
    else:
        with open(path) as fh:
            text = fh.read()
    rows = []; labels = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",") if "," in line else line.split()
        rows.append([float(v) for v in parts[:-1]])
        labels.append(parts[-1].strip())
    return np.array(rows, dtype=np.float64), np.array(labels)
