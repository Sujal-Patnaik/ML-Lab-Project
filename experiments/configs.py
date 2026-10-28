# Model zoo: the 12 models of paper 1 plus our adaptive model, each written as the
# SupportVectorForest settings that define it. n_trees and seed are added when a model is built.
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from svf import SupportVectorForest  # noqa: E402

PAPER_MODELS = {
    # standard random forest family: one bootstrap per tree
    "RaF":       dict(splitter="axis", bootstrap="tree"),
    "MPRaF-T":   dict(splitter="mpsvm", variant="T", bootstrap="tree"),
    "MPRaF-P":   dict(splitter="mpsvm", variant="P", bootstrap="tree"),
    "MPRaF-N":   dict(splitter="mpsvm", variant="N", bootstrap="tree"),
    "RaF-PCA":   dict(splitter="pca", bootstrap="tree"),
    "RaF-LDA":   dict(splitter="lda", bootstrap="tree"),
    # double random forest family: bootstrap at every big node
    "DRaF":      dict(splitter="axis", bootstrap="node"),
    "MPDRaF-T":  dict(splitter="mpsvm", variant="T", bootstrap="node"),
    "MPDRaF-P":  dict(splitter="mpsvm", variant="P", bootstrap="node"),
    "MPDRaF-N":  dict(splitter="mpsvm", variant="N", bootstrap="node"),
    "DRaF-PCA":  dict(splitter="pca", bootstrap="node"),
    "DRaF-LDA":  dict(splitter="lda", bootstrap="node"),
}

OUR_MODELS = {
    # novelty: per-node choice among the axis, LDA and MPSVM-P splitters
    "SVF-Adaptive": dict(splitter="adaptive", candidates=["axis", "lda", "mpsvm"], variant="P",
                         bootstrap="node"),
}

ALL_MODELS = {**PAPER_MODELS, **OUR_MODELS}


def make_model(name: str, n_trees: int = 50, seed: int = 0, **overrides) -> SupportVectorForest:
    # An unfitted forest is built for a model name; overrides (e.g. mtry=5) are applied last.
    if name not in ALL_MODELS:
        raise KeyError(f"unknown model '{name}', valid names: {list(ALL_MODELS)}")
    params = dict(ALL_MODELS[name])
    params.update(n_trees=n_trees, seed=seed)
    params.update(overrides)
    return SupportVectorForest(**params)
