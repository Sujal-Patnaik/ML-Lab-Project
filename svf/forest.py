# SupportVectorForest: the forest model (scikit-learn compatible).
#     model = SupportVectorForest(n_trees=50, splitter='mpsvm', variant='P', bootstrap='node')
#     model.fit(X_train, y_train).predict(X_test)
# The input is validated and the labels are encoded once, one independent seed is spawned
# per tree, L trees are grown, and the prediction is made by majority vote as in the paper.
# __init__ only stores its arguments, so clone(), cross_val_score() and Pipeline work.
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .tree import DoubleRandomTree, grow_tree


# Parameters:
#   n_trees    : number of trees L (paper: 50)
#   splitter   : 'axis' | 'mpsvm' | 'pca' | 'lda' | 'adaptive'
#   variant    : MPSVM regularization 'T' | 'P' | 'N';  delta : Tikhonov term
#   eps        : ridge for LDA / Bhattacharyya;  candidates : splitter names for 'adaptive'
#   mtry       : features per node (None -> round(sqrt(n)));  minleaf : leaf size
#   bootstrap  : 'node' (DRaF family), 'tree' (RaF family), 'none';  boot_frac : 0.1 in the paper
#   max_depth  : None or int;  seed : the only source of randomness of the whole forest
# Fitted attributes: classes_ (C,), n_features_in_, trees_ (list of L DoubleRandomTree).
class SupportVectorForest(ClassifierMixin, BaseEstimator):

    def __init__(self, n_trees: int = 50, splitter: str = "mpsvm", variant: str = "T",
                 delta: float = 1e-4, eps: float = 1e-6, candidates: Optional[Sequence[str]] = None,
                 mtry: Optional[int] = None, minleaf: int = 1, bootstrap: str = "node",
                 boot_frac: float = 0.1, max_depth: Optional[int] = None, seed: int = 0):
        self.n_trees, self.splitter, self.variant = n_trees, splitter, variant
        self.delta, self.eps, self.candidates = delta, eps, candidates
        self.mtry, self.minleaf, self.bootstrap = mtry, minleaf, bootstrap
        self.boot_frac, self.max_depth, self.seed = boot_frac, max_depth, seed

    def _tree_params(self) -> dict:
        # Plain dict that describes one tree, passed to grow_tree().
        candidates = list(self.candidates) if self.candidates is not None else None
        splitter_params = {'variant': self.variant, 'delta': self.delta, 'eps': self.eps,
                           'candidates': candidates}
        return {'splitter': self.splitter, 'splitter_params': splitter_params,
                'mtry': self.mtry, 'minleaf': self.minleaf, 'bootstrap': self.bootstrap,
                'boot_frac': self.boot_frac, 'max_depth': self.max_depth}

    def fit(self, X, y) -> "SupportVectorForest":
        # X (N, n) is made contiguous float64 and y (N,) is encoded to 0..C-1.
        X, y = check_X_y(X, y)
        X = np.ascontiguousarray(X, dtype=np.float64)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        y_enc = np.ascontiguousarray(y_enc, dtype=np.int64)
        C = len(self.classes_)
        if(C<2):
            raise ValueError("need at least 2 classes in y to fit a classifier")
        self.n_features_in_ = X.shape[1]
        # L independent, reproducible random streams are spawned; tree i always gets seeds[i]
        seeds = np.random.SeedSequence(self.seed).spawn(self.n_trees)
        tree_params = self._tree_params()
        self.trees_ = []
        for i in range(self.n_trees):
            self.trees_.append(grow_tree(X, y_enc, C, tree_params, seeds[i]))
        return self

    def _check_X(self, X) -> np.ndarray:
        # Prediction input is checked against the fitted model.
        check_is_fitted(self, "trees_")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, model was fitted with {self.n_features_in_}")
        return np.asarray(X, dtype=np.float64)

    def tree_predictions(self, X) -> np.ndarray:
        # Encoded prediction of every tree -> (L, m); row t comes from tree t.
        X = self._check_X(X)
        L = len(self.trees_)
        P = np.zeros((L, X.shape[0]), dtype=np.int64)
        for t in range(L):
            P[t] = self.trees_[t].predict(X)
        return P

    def predict(self, X) -> np.ndarray:
        # Majority vote: votes (m, C) are counted over the L trees, the class with most votes
        # is taken (first class on ties) and mapped back to the original labels -> (m,).
        P = self.tree_predictions(X)
        L, m = P.shape
        C = len(self.classes_)
        votes = np.zeros((m, C), dtype=np.int64)
        for t in range(L):
            for i in range(m):
                votes[i, P[t,i]] += 1   # each tree gives one vote per sample
        pred = np.zeros(m, dtype=np.int64)
        for i in range(m):
            best = 0
            for c in range(1, C):
                if(votes[i,c]>votes[i,best]):   # strict: first class wins ties
                    best = c
            pred[i] = best
        return self.classes_[pred]

    def predict_proba(self, X) -> np.ndarray:
        # Soft output (m, C): mean of the leaf probabilities of all trees.
        X = self._check_X(X)
        proba = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        for t in self.trees_:
            proba += t.predict_proba(X)
        return proba / len(self.trees_)

    def tree_sizes(self) -> np.ndarray:
        # Number of nodes of each tree -> (L,). Double forests grow bigger trees than RaF.
        check_is_fitted(self, "trees_")
        return np.array([t.n_nodes() for t in self.trees_])
