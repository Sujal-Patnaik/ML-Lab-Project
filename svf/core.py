# Core building blocks used by every other module:
#   - the split / node data types,
#   - Gini impurity and the Gini gain of a split,
#   - the axis-parallel threshold search.
#
# Notation used throughout the code:
#   X  : (N, n) float64 training matrix, N samples and n features
#   y  : (N,) int64 labels, already encoded to 0..C-1 by the forest
#   C  : number of classes in the whole training set
#   k  : number of features drawn at a node (mtry, default round(sqrt(n)))
#   Xs : (m, k) data seen by a splitter = bootstrap rows of the node x the k drawn columns
#   ys : (m,) labels of those rows
# Every split is stored as a hyperplane: a sample x goes LEFT if x[feats] @ w - b <= 0.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------- data types

# Result returned by every splitter.
#   w    : (k,) normal of the hyperplane, in the space of the k drawn features
#   b    : offset, rule is  x_local @ w - b <= 0  -> left
#   gain : Gini decrease measured on the bootstrap sample the splitter saw
#   kind : 'axis', 'mpsvm', 'pca' or 'lda'
@dataclass
class SplitResult:
    w: np.ndarray
    b: float
    gain: float
    kind: str


# Marker returned by MPSVM variant 'P' when P or Q is rank deficient.
# The node and its whole subtree are then split with axis-parallel cuts.
class _FallbackToAxis:

    def __repr__(self) -> str:
        return "FALLBACK_TO_AXIS"


FALLBACK_TO_AXIS = _FallbackToAxis()


# Best 1-D cut found over several columns: Gini gain, local column index and threshold.
@dataclass
class AxisSplit:
    gain: float
    column: int
    threshold: float


# One tree node. A node is a leaf when value is set.
#   internal node : feats (k,), w (k,), b, kind, left, right
#   leaf          : value (C,) = class proportions of the samples that reached it
#   both          : n_samples (original samples at the node) and depth
@dataclass
class Node:
    n_samples: int
    depth: int
    value: Optional[np.ndarray] = None
    feats: Optional[np.ndarray] = None
    w: Optional[np.ndarray] = None
    b: float = 0.0
    kind: str = ""
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


# ---------------------------------------------------------------- Gini impurity
# Gini impurity of a set with class proportions p_c = n_c / n:
#     G(S) = 1 - sum_c p_c^2
# Gain of a split into S_L and S_R:
#     gain = G(S) - (n_L/n) G(S_L) - (n_R/n) G(S_R)
# A larger gain means purer children.


def class_counts(y: np.ndarray, C: int) -> np.ndarray:
    # Number of samples of each class in y is counted -> (C,) float array.
    # A label outside 0..C-1 is reported as an error.
    counts = np.zeros(C, dtype=np.float64)
    for cls in y:
        if (cls<C and cls>=0):
            counts[cls]+=1
        else:
            raise ValueError(f"label {cls} outside 0..{C-1}")
    return counts


def gini(counts: np.ndarray) -> float:
    # Gini impurity is computed from the (C,) count vector; an empty set is given 0.
    GINI = 1.0
    s = sum(counts)
    if(s==0):
        return 0.0
    for ele in counts:
        GINI -= ((ele)/(s))**2
    return GINI

def gain_from_counts(parent: np.ndarray, left: np.ndarray) -> float:
    # Gini gain of a split, given the (C,) counts of the parent and of the left child.
    # The right child counts are obtained as parent - left.
    # A split that leaves one side empty is given gain -inf, so it is never chosen.
    gain = gini(parent)
    C = len(parent)
    n = sum(parent); n_L = sum(left); n_R = n-n_L
    if(n_L==0 or n_R==0):
        return -np.inf
    w_L = float(n_L/n); w_R = float(n_R/n)
    right = np.zeros(C, dtype=np.float64)
    for i in range(C):
        right[i] = parent[i]-left[i]
    gain -= w_L * gini(left)
    gain -= w_R * gini(right)
    return gain
def split_gain(y: np.ndarray, go_left: np.ndarray, C: int) -> float:
    # Gini gain of a split given as a boolean mask go_left (m,) over the labels y (m,).
    parent = class_counts(y, C)
    left = np.zeros(C, dtype=np.float64)
    for i in range(len(y)):
        if go_left[i]:
            left[y[i]] += 1
    return gain_from_counts(parent, left)


# ---------------------------------------------------------------- axis-parallel search
# Used by RaF / DRaF, inside the PCA / LDA rotation splits, and as the MPSVM-P fallback.
# For one column z (m,): the samples are sorted by z and every cut between two consecutive
# DIFFERENT values is scored. Moving the cut one position moves a single sample to the left,
# so the left counts are updated with one increment per position.
# The threshold is the midpoint of the two values and  z <= t  goes left.


def best_threshold_1d(z: np.ndarray, y: np.ndarray, C: int) -> Optional[Tuple[float, float]]:
    # Best (gain, threshold) on one column z (m,), or None if all values of z are equal.
    m = len(z)
    order = sorted(range(m), key=lambda i:z[i])
    zs = [z[order[k]] for k in range(m)]
    ys = [y[order[k]] for k in range(m)]
    total = class_counts(y,C)
    left = np.zeros(C, dtype=np.float64)
    best_gain = -np.inf; best_threshold = None
    for i in range(m-1):
        left[ys[i]] += 1
        if(zs[i]==zs[i+1]):
            continue            # equal values cannot be separated by a threshold
        g = gain_from_counts(total,left)
        if g>best_gain:         # strict: the first best cut is kept on ties
            thr = 0.5*(zs[i] + zs[i+1])
            best_gain, best_threshold = g,thr
    if(best_threshold is None):
        return None
    return best_gain,best_threshold


def best_axis_split(Z: np.ndarray, y: np.ndarray, C: int) -> Optional[AxisSplit]:
    # Every column of Z (m, k) is searched; the column with the largest gain is returned
    # (the earlier column wins ties). None is returned if no column can be cut.
    m,k = Z.shape
    best_gain = -np.inf; best_threshold = None; col = None
    for j in range(k):
        t = best_threshold_1d(Z[:,j],y,C)
        if t is not None:
            g,thr = t[0],t[1]
            if(g>best_gain):
                best_gain,best_threshold = g,thr
                col = j
    if(best_threshold is not None):
        A = AxisSplit(gain=best_gain,column=col,threshold=best_threshold)
        return A
    return None
