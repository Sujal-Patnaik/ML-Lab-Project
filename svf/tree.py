# One tree of the forest (paper 1, Algorithms 3, 5 and 6).
# At a node holding the ORIGINAL sample indices idx (N_d of them):
#   0. the node is made a leaf if it is pure, if N_d <= minleaf, or if max_depth is reached
#   1. 'node' mode: if N_d > boot_frac * N, a bootstrap of size N_d is drawn from idx
#   2. k = mtry features are drawn without replacement
#   3. the splitter chooses a rule on Xs = X[boot][:, feats]  (the bootstrap only CHOOSES)
#   4. the ORIGINAL idx are routed with  X[idx][:, feats] @ w - b <= 0
#   5. the left subtree is grown first, then the right one
# Bootstrap modes:
#   'node' : DRaF family - every tree starts from all N samples, bootstrap at each big node
#   'tree' : RaF family  - one bootstrap of size N at the root, no node bootstrap
#   'none' : no bootstrap at all
# All randomness of a tree comes from its own rng, always in the order root bootstrap
# ('tree' mode), then per node: node bootstrap ('node' mode), feature draw.
from __future__ import annotations

from collections import Counter
from typing import Optional, Tuple

import numpy as np

from .splitters import AxisSplitter, Splitter, make_splitter
from .core import FALLBACK_TO_AXIS, Node, SplitResult, class_counts
_AXIS = AxisSplitter()


# A single (double) random tree.
#   splitter  : Splitter object used at every node
#   mtry      : features drawn per node (None -> max(1, round(sqrt(n))))
#   minleaf   : a node with N_d <= minleaf becomes a leaf
#   bootstrap : 'node' | 'tree' | 'none';  boot_frac : node-bootstrap threshold (paper: 0.1)
#   rng       : np.random.Generator, the only source of randomness of the tree
# Fitted attributes: root_, N_ (training size), n_features_, n_classes_, k_ (mtry used).
class DoubleRandomTree:

    def __init__(self, splitter: Splitter, mtry: Optional[int] = None, minleaf: int = 1,
                 bootstrap: str = "node", boot_frac: float = 0.1,
                 max_depth: Optional[int] = None, rng: Optional[np.random.Generator] = None):
        if bootstrap not in ("node", "tree", "none"):
            raise ValueError("bootstrap must be 'node', 'tree' or 'none'")
        self.splitter, self.mtry, self.minleaf = splitter, mtry, minleaf
        self.bootstrap, self.boot_frac, self.max_depth = bootstrap, boot_frac, max_depth
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------ training
    def fit(self, X: np.ndarray, y: np.ndarray, C: int) -> "DoubleRandomTree":
        # X (N, n), y (N,) encoded labels; the whole tree is grown from the root.
        self.N_, self.n_features_ = X.shape
        self.n_classes_ = C
        self.k_ = self._resolve_mtry(self.n_features_)
        idx = self._root_indices()
        self.root_ = self._grow(X, y, idx, depth=0, axis_only=False)
        return self

    def _resolve_mtry(self, n: int) -> int:
        # Number of features drawn per node: mtry if given (1 <= mtry <= n), else round(sqrt(n)).
        if self.mtry is None:
            return max(1,int(round(np.sqrt(n))))
        if(self.mtry>=1 and self.mtry<=n):
            return int(self.mtry)
        else:
            raise ValueError(f"mtry={self.mtry} must be between 1 and n_features={n}")


    def _root_indices(self) -> np.ndarray:
        # Indices entering the root: a bootstrap of size N in 'tree' mode (duplicates kept),
        # otherwise all N samples.
        if(self.bootstrap == "tree"):
            return self.rng.choice(self.N_, size=self.N_,replace=True)
        else:
            return np.arange(self.N_)


    def _should_stop(self, y_node: np.ndarray, depth: int) -> bool:
        # Step 0: stopping rules of a node.
        if(len(y_node)<=self.minleaf):
            return True  # the node has minleaf samples or fewer
        if(min(y_node)==max(y_node)):
            return True # Only one class is present so return True
        if(self.max_depth is not None and depth >= self.max_depth):
            return True # Growth should stop and the tree should not grow more than max_depth
        return False

    def _node_sample(self, idx: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Step 1: indices used to CHOOSE the split. In 'node' mode a big node
        # (N_d > boot_frac * N) is bootstrapped; a bootstrap with a single class cannot be
        # split, so idx is used instead. In every other case idx itself is returned.
        if(self.bootstrap == "node" and len(idx) > self.boot_frac * self.N_):
            boot = self.rng.choice(idx,size=len(idx),replace=True)
            if(min(y[boot])==max(y[boot])):
                return idx # As only a single class is there
            return boot
        return idx

    def _draw_features(self) -> np.ndarray:
        # Step 2: k_ feature indices drawn without replacement -> (k,).
        return self.rng.choice(self.n_features_,size=self.k_,replace=False)
    def _find_split(self, Xs: np.ndarray, ys: np.ndarray,
                    axis_only: bool) -> Tuple[Optional[SplitResult], bool]:
        # Step 3: a rule is requested from the splitter for Xs (m, k), ys (m,).
        # Returns (result, axis_only flag for the children):
        #   - axis_only set by an ancestor      -> axis split, children stay axis-only
        #   - splitter gives FALLBACK_TO_AXIS   -> axis split, children become axis-only
        #   - non-axis splitter gives None      -> one axis split as a rescue at this node only
        if(axis_only):
            return _AXIS.split(Xs, ys, self.n_classes_), True
        res = self.splitter.split(Xs,ys,self.n_classes_)
        if(res is FALLBACK_TO_AXIS):
            return _AXIS.split(Xs, ys, self.n_classes_), True
        if res is None and not isinstance(self.splitter,AxisSplitter):
            res = _AXIS.split(Xs, ys, self.n_classes_) # one-off rescue, children are NOT forced to axis
        return res,False
    def _make_leaf(self, y_node: np.ndarray, depth: int) -> Node:
        # Leaf value (C,) = class proportions of the ORIGINAL samples at the node.
        counts = class_counts(y_node,self.n_classes_)
        value = counts / len(y_node)
        return Node(n_samples=len(y_node), depth=depth, value=value)
    def _grow(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, depth: int,
              axis_only: bool) -> Node:
        # Recursive construction of the subtree for the original indices idx (steps 0-5).
        # X and y are the full training data and are only indexed, never copied.
        y_node = y[idx]
        if(self._should_stop(y_node, depth)):
            return self._make_leaf(y_node, depth)
        boot = self._node_sample(idx, y)
        feats = self._draw_features()
        res, child_axis = self._find_split(X[np.ix_(boot, feats)], y[boot], axis_only)
        if res is None:
            return self._make_leaf(y_node, depth)
        go_left = X[np.ix_(idx, feats)] @ res.w - res.b <= 0      # (N_d,) routing of the original data
        if(go_left.all() or not go_left.any()):
            return self._make_leaf(y_node, depth) # rule chosen on the bootstrap does not split the original data
        node = Node(n_samples=len(idx), depth=depth, feats=feats, w=res.w, b=res.b, kind=res.kind)
        node.left = self._grow(X, y, idx[go_left], depth+1, child_axis)   # left subtree first
        node.right = self._grow(X, y, idx[~go_left], depth+1, child_axis)
        return node

    # ------------------------------------------------------------------ prediction
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Class probabilities (m, C) for X (m, n): sets of row indices are routed down the
        # tree together and each row receives the value of the leaf it reaches.
        m = X.shape[0]
        out = np.zeros((m, self.n_classes_), dtype=np.float64)

        def visit(node, rows):
            if(len(rows)==0):
                return
            if node.is_leaf:
                out[rows] = node.value
                return
            go_left = X[np.ix_(rows, node.feats)] @ node.w - node.b <= 0
            visit(node.left, rows[go_left])
            visit(node.right, rows[~go_left])

        visit(self.root_, np.arange(m))
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Encoded class (m,) = argmax of the leaf probabilities.
        return self.predict_proba(X).argmax(1)

    # ------------------------------------------------------------------ statistics
    # The tree is walked with an explicit stack in each of these.
    def n_nodes(self) -> int:
        # Total number of nodes (internal + leaves).
        count = 0
        stack = [self.root_]
        while stack:
            node = stack.pop()
            count += 1
            if not node.is_leaf:
                stack.append(node.left); stack.append(node.right)
        return count

    def n_leaves(self) -> int:
        # Number of leaves; for a binary tree this equals (n_nodes + 1) // 2.
        count = 0
        stack = [self.root_]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                count += 1
            else:
                stack.append(node.left); stack.append(node.right)
        return count

    def depth(self) -> int:
        # Largest node depth (root alone -> 0).
        d = 0
        stack = [self.root_]
        while stack:
            node = stack.pop()
            if(node.depth>d):
                d = node.depth
            if not node.is_leaf:
                stack.append(node.left); stack.append(node.right)
        return d

    def split_kinds(self) -> Counter:
        # How many internal nodes used each split kind, e.g. {'mpsvm': 30, 'axis': 12}.
        kinds = Counter()
        stack = [self.root_]
        while stack:
            node = stack.pop()
            if not node.is_leaf:
                kinds[node.kind] += 1
                stack.append(node.left); stack.append(node.right)
        return kinds


def grow_tree(X: np.ndarray, y: np.ndarray, C: int, tree_params: dict,
              seed: np.random.SeedSequence) -> DoubleRandomTree:
    # One tree is built and fitted from a plain parameter dict and its own seed.
    # tree_params example: {'splitter': 'mpsvm', 'splitter_params': {'variant': 'T'},
    #                       'mtry': None, 'minleaf': 1, 'bootstrap': 'node', 'boot_frac': 0.1,
    #                       'max_depth': None}
    p = dict(tree_params)   # copy: the caller's dict is shared by all trees
    p['splitter_params'] = dict(p.get('splitter_params', {}))
    splitter = make_splitter(p.pop('splitter'), **p.pop('splitter_params'))
    rng = np.random.default_rng(seed)
    return DoubleRandomTree(splitter, rng=rng, **p).fit(X, y, C)
