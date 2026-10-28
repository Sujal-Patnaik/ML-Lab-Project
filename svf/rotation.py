# Rotation splits: RaF-PCA / DRaF-PCA and RaF-LDA / DRaF-LDA (paper 1, Section 3.2).
# At a node a (k, k) rotation V is computed from the bootstrap data, the drawn columns are
# rotated as Z = Xs @ V, and the ordinary axis search is run on Z. If the best cut is
# column j at threshold t, the split is stored as  w = V[:, j],  b = t,  so a rotation
# split is an oblique split whose direction is a PCA / LDA axis.
#   PCA : eigenvectors of the scatter matrix of the centred data (all k are kept)
#   LDA : generalized eigenvectors of  Sb a = lambda (Sw + eps*I) a
#         Sw = sum_c sum_{i in c} (x_i - mu_c)(x_i - mu_c)^T   (within-class scatter)
#         Sb = sum_c m_c (mu_c - mu)(mu_c - mu)^T               (between-class scatter)
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import eigh

from .core import SplitResult, best_axis_split


def pca_basis(Xs: np.ndarray) -> np.ndarray:
    # V (k, k): eigenvectors of Xc^T Xc, where Xc (m, k) is the centred node data.
    m,k = Xs.shape
    mu = np.zeros(k,dtype=np.float64)
    for i in range(m):
        mu += Xs[i,:]
    mu/=m
    Xc = Xs - mu
    eigenval,evec = np.linalg.eigh(Xc.T @ Xc)
    return evec
def lda_basis(Xs: np.ndarray, ys: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # V (k, k): generalized eigenvectors of Sb and Sw + eps*I, both (k, k).
    # Only C-1 eigenvalues are non-zero, but all k directions are kept as rotation axes.
    m,k = Xs.shape
    mu = Xs.mean(0)
    Sw = np.zeros((k,k),dtype=np.float64)
    Sb = np.zeros((k,k),dtype=np.float64)
    classes = np.unique(ys)
    for c in classes:
        Xc = Xs[ys==c]
        row,col = Xc.shape
        mc = Xc.mean(0)
        Sw += (Xc-mc).T @ (Xc-mc)
        Sb += row * np.outer(mc-mu,mc-mu)
    I = np.zeros((k,k),dtype=np.float64)
    for i in range(k):
        I[i,i]=1
    eigenval,evec = eigh(Sb,Sw + eps*I)
    return evec
def rotation_split(Xs: np.ndarray, ys: np.ndarray, C: int, kind: str = "pca",
                   eps: float = 1e-6) -> Optional[SplitResult]:
    # The node data is rotated and searched axis by axis; the chosen axis V[:, j] becomes w.
    # None is returned when the axis search finds no cut.
    m,k = Xs.shape
    V = pca_basis(Xs) if kind=='pca' else lda_basis(Xs,ys,eps)
    r = best_axis_split(Xs @ V, ys, C)
    if r is None:
        return None
    gain = r.gain; col = r.column; thr = r.threshold
    res = SplitResult(V[:,col],thr,gain,kind)
    return res
