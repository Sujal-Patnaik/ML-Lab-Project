# Paper 2 (Han & Joe 2024) ideas, written from scratch and applied without data leakage:
#   - preprocessing: StandardScaler and an optional global PCA
#   - SMOTE-ENN class balancing
#   - SVFPipeline: preprocessing -> (SMOTE-ENN) -> forest, all fitted on the training data only
#   - seed averaging (what paper 2 calls "Stochastic Weighted Averaging")
from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------- preprocessing
# The global PCA here is applied ONCE to the whole data before the forest, as in paper 2.
# It is different from the node-level PCA rotation of DRaF-PCA (rotation.py).


def make_preprocessor(scale: bool = True, pca: Optional[Union[int, float]] = None,
                      random_state: int = 0) -> Pipeline:
    # An unfitted sklearn Pipeline is built with the steps 'scale' and/or 'pca'.
    #   scale : z-scores per feature (MPSVM and LDA / PCA splits are scale sensitive)
    #   pca   : None -> no PCA; int -> number of components; float in (0, 1) -> fraction
    #           of variance to keep
    # When both are off an identity step is used, so fit_transform / transform always work.
    steps = []
    if scale:
        steps.append(("scale", StandardScaler()))
    if pca is not None:
        steps.append(("pca", PCA(n_components=pca, random_state=random_state)))
    if(len(steps)==0):
        steps.append(("identity", FunctionTransformer()))
    return Pipeline(steps)


# ---------------------------------------------------------------- SMOTE-ENN
# SMOTE (oversampling): a synthetic point is placed on the segment between a minority sample
# x_i and one of its k nearest same-class neighbours x_nn:
#     x_new = x_i + u (x_nn - x_i),   u ~ U(0, 1)
# ENN (cleaning): a sample is removed when its k nearest neighbours (itself excluded)
# disagree with its label, so noisy and borderline points (also bad synthetic ones) go away.
# Here SMOTE is applied first and ENN second, and only ever on the training data.


def smote(X: np.ndarray, y: np.ndarray, k: int = 5,
          rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    # Every non-majority class is oversampled up to the majority count.
    # X (N, n), y (N,) -> (X_res, y_res) with the original rows first, then the synthetic rows.
    if rng is None:
        rng = np.random.default_rng()
    classes = np.unique(y)
    counts = {}
    for c in classes:
        counts[c] = 0
    for label in y:
        counts[label] += 1
    target = max(counts.values())
    X_blocks = [X]; y_blocks = [y]
    for c in classes:
        n_c = counts[c]
        if(n_c>=target):
            continue
        Xc = X[y==c]                         # (n_c, n) rows of this class
        if(n_c<2):
            warnings.warn(f"class {c} has {n_c} sample, SMOTE cannot interpolate - skipped")
            continue
        kk = min(k, n_c-1)
        nn = NearestNeighbors(n_neighbors=kk+1).fit(Xc)
        nbrs = nn.kneighbors(Xc, return_distance=False)[:,1:]   # column 0 is the point itself
        g = target - n_c                     # number of synthetic points for this class
        i = rng.integers(0, n_c, g)      # draw order i, j, u (fixed, for reproducibility)
        jj = rng.integers(0, kk, g)
        u = rng.random(g)
        new = np.zeros((g, X.shape[1]), dtype=np.float64)
        for t in range(g):
            a = Xc[i[t]]; nb = Xc[nbrs[i[t], jj[t]]]
            new[t] = a + u[t] * (nb - a)   # point on the segment between a and its neighbour
        X_blocks.append(new); y_blocks.append(np.full(g, c, dtype=y.dtype))
    return np.vstack(X_blocks), np.concatenate(y_blocks)


def enn_mask(X: np.ndarray, y: np.ndarray, k: int = 3, kind: str = "all") -> np.ndarray:
    # Boolean mask (N,) of the samples ENN KEEPS.
    #   kind 'all'  : kept only if all k neighbours have the same label
    #   kind 'mode' : kept if the majority of the k neighbours have the same label
    N = len(y)
    kk = min(k, N-1)
    if(kk<1):
        return np.ones(N, dtype=bool)
    nbrs = NearestNeighbors(n_neighbors=kk+1).fit(X).kneighbors(X, return_distance=False)[:,1:]   # (N, kk)
    if kind not in ("all", "mode"):
        raise ValueError("kind must be 'all' or 'mode'")
    keep = np.zeros(N, dtype=bool)
    for i in range(N):
        same = 0
        for j in range(kk):
            if(y[nbrs[i,j]]==y[i]):
                same += 1
        if kind=="all":
            keep[i] = (same==kk)
        else:
            keep[i] = (same > kk/2)
    return keep


# SMOTE followed by ENN.
#   k_smote : SMOTE neighbours;  k_enn : ENN neighbours;  kind : ENN rule;  random_state : seed
class SMOTEENN:

    def __init__(self, k_smote: int = 5, k_enn: int = 3, kind: str = "all", random_state: int = 0):
        self.k_smote, self.k_enn, self.kind, self.random_state = k_smote, k_enn, kind, random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # The balanced and cleaned training set is returned. If ENN would remove every sample
        # of a class, that class is kept as it is (a class never seen cannot be learned).
        rng = np.random.default_rng(self.random_state)
        Xr, yr = smote(X, y, self.k_smote, rng)
        keep = enn_mask(Xr, yr, self.k_enn, self.kind)
        for c in np.unique(yr):
            if not keep[yr==c].any():
                keep[yr==c] = True   # never delete a whole class
        return Xr[keep], yr[keep]


# ---------------------------------------------------------------- leakage-free pipeline
# fit     : the preprocessor is fitted on the training data, the training data are resampled,
#           and the model is fitted on the result.
# predict : the test data are only transformed (never resampled, never refitted).
# Paper 2 fitted the preprocessing on train + test together and resampled before CV;
# here everything is done inside fit, so cross_val_score redoes it in every fold.
#   model : unfitted classifier;  scale, pca : see make_preprocessor
#   resample : apply SMOTE-ENN;  random_state : seed for PCA and SMOTE-ENN
class SVFPipeline(ClassifierMixin, BaseEstimator):

    def __init__(self, model=None, scale: bool = True, pca: Optional[Union[int, float]] = None,
                 resample: bool = False, random_state: int = 0):
        self.model, self.scale, self.pca = model, scale, pca
        self.resample, self.random_state = resample, random_state

    def fit(self, X, y) -> "SVFPipeline":
        # Resampling is done AFTER scaling / PCA, so SMOTE and ENN use the same geometry
        # the model sees.
        self.pre_ = make_preprocessor(self.scale, self.pca, self.random_state)
        Xp = self.pre_.fit_transform(X)       # fitted on training data only
        y = np.asarray(y)
        if self.resample:
            Xp, y = SMOTEENN(random_state=self.random_state).fit_resample(Xp, y)
        self.model_ = clone(self.model).fit(Xp, y)
        self.classes_ = self.model_.classes_
        self.n_train_resampled_ = len(y)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(self.pre_.transform(X))

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(self.pre_.transform(X))


# ---------------------------------------------------------------- seed averaging
# Paper 2 retrains the same model with several random seeds and averages the predicted
# probabilities. That is seed (ensemble) averaging, not SWA, which averages the WEIGHTS along
# an SGD path. For a forest, S forests of L trees behave much like one forest of S*L trees.
#   estimator  : unfitted classifier with a seed parameter and predict_proba
#   n_seeds    : number of copies S;  seed_param : name of that parameter;  base_seed : master seed
class SeedAveragedClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, estimator=None, n_seeds: int = 5, seed_param: str = "seed", base_seed: int = 0):
        self.estimator, self.n_seeds = estimator, n_seeds
        self.seed_param, self.base_seed = seed_param, base_seed

    def fit(self, X, y) -> "SeedAveragedClassifier":
        # S different 32-bit seeds are generated from the master seed and one copy is fitted
        # per seed.
        self.seeds_ = np.random.SeedSequence(self.base_seed).generate_state(self.n_seeds)
        self.estimators_ = []
        for s in self.seeds_:
            est = clone(self.estimator).set_params(**{self.seed_param: int(s)})
            self.estimators_.append(est.fit(X, y))
        self.classes_ = self.estimators_[0].classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        # Mean of the S probability matrices, each (m, C).
        check_is_fitted(self, "estimators_")
        proba = None
        for est in self.estimators_:
            P = est.predict_proba(X)
            proba = P if proba is None else proba + P
        return proba / len(self.estimators_)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]
