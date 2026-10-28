# Splitter classes: one common interface split(Xs, ys, C) over all split algorithms.
# The tree only calls splitter.split() and receives a SplitResult, None or FALLBACK_TO_AXIS,
# so the same tree class is used for every model of the paper:
#     RaF / DRaF           -> AxisSplitter
#     MPRaF-* / MPDRaF-*   -> MPSVMSplitter('T' | 'P' | 'N')
#     RaF-PCA / DRaF-PCA   -> RotationSplitter('pca')
#     RaF-LDA / DRaF-LDA   -> RotationSplitter('lda')
#     SVF-Adaptive (ours)  -> AdaptiveSplitter([axis, lda, mpsvm-P])
# The RaF and DRaF families differ only in the tree's bootstrap mode ('tree' or 'node').
from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from .core import FALLBACK_TO_AXIS, SplitResult, _FallbackToAxis, best_axis_split
from .mpsvm import mpsvm_split
from .rotation import rotation_split

SplitOutcome = Union[SplitResult, None, _FallbackToAxis]


# Base class; name is a short label such as 'axis', 'mpsvm-T' or 'pca'.
class Splitter:
    name = "base"

    def split(self, Xs: np.ndarray, ys: np.ndarray, C: int) -> SplitOutcome:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"


# Axis-parallel (CART style) split.
class AxisSplitter(Splitter):
    name = "axis"

    def split(self, Xs: np.ndarray, ys: np.ndarray, C: int) -> SplitOutcome:
        # The best axis cut on column j is stored as a one-hot w = e_j (k,) with b = threshold,
        # so axis and oblique nodes are routed by the same rule  x[feats] @ w - b <= 0.
        r = best_axis_split(Xs,ys,C)
        if(r is None):
            return None
        k = Xs.shape[1]
        w = np.zeros(k); w[r.column] = 1.0
        return SplitResult(w=w,b=r.threshold, gain=r.gain, kind="axis")


# Oblique MPSVM split with regularization 'T', 'P' or 'N'.
class MPSVMSplitter(Splitter):

    def __init__(self, variant: str = "T", delta: float = 1e-4):
        if variant not in ("T", "P", "N"):
            raise ValueError("variant must be 'T', 'P' or 'N'")
        self.variant, self.delta = variant, delta
        self.name = f"mpsvm-{variant}"

    def split(self, Xs: np.ndarray, ys: np.ndarray, C: int) -> SplitOutcome:
        # FALLBACK_TO_AXIS (variant 'P' only) is passed on and handled by the tree.
        return mpsvm_split(Xs,ys,C,self.variant,self.delta)

# PCA or LDA rotation followed by an axis search.
class RotationSplitter(Splitter):

    def __init__(self, kind: str = "pca", eps: float = 1e-6):
        if kind not in ("pca", "lda"):
            raise ValueError("kind must be 'pca' or 'lda'")
        self.kind, self.eps = kind, eps
        self.name = kind

    def split(self, Xs: np.ndarray, ys: np.ndarray, C: int) -> SplitOutcome:
        return rotation_split(Xs, ys, C, kind=self.kind, eps=self.eps)

# Our novelty: every candidate splitter is tried at each node on the SAME bootstrap
# (Xs, ys) and the split with the largest Gini gain is kept (the first candidate wins ties).
# Candidates that return None or FALLBACK_TO_AXIS are ignored. The winning result keeps its
# own kind, so it can be seen which family was chosen at each node.
class AdaptiveSplitter(Splitter):

    def __init__(self, candidates: Sequence[Splitter]):
        if not candidates:
            raise ValueError("need at least one candidate splitter")
        self.candidates = list(candidates)
        self.name = "adaptive(" + ",".join(c.name for c in self.candidates) + ")"

    def split(self, Xs: np.ndarray, ys: np.ndarray, C: int) -> SplitOutcome:
        best = None
        for c in self.candidates:
            r = c.split(Xs,ys,C)
            if isinstance(r,SplitResult) and (best is None or r.gain>best.gain):
                best = r
        return best

def make_splitter(name: str, **params) -> Splitter:
    # A splitter is built from its short name: 'axis' | 'mpsvm' | 'pca' | 'lda' | 'adaptive'.
    # Parameters not used by the chosen splitter are ignored, so the forest can pass all of
    # them. For 'adaptive' the default candidates are ['axis', 'lda', 'mpsvm'].
    variant = params.get("variant", "T")
    delta = params.get("delta", 1e-4)
    eps = params.get("eps", 1e-6)

    if name == "axis":
        return AxisSplitter()
    if name == "mpsvm":
        return MPSVMSplitter(variant, delta)
    if name in ("pca", "lda"):
        return RotationSplitter(name, eps)
    if name == "adaptive":
        names = params.get("candidates") or ["axis", "lda", "mpsvm"]
        members = [make_splitter(n, variant=variant, delta=delta, eps=eps) for n in names]
        return AdaptiveSplitter(members)
    raise ValueError(f"unknown splitter '{name}'")
