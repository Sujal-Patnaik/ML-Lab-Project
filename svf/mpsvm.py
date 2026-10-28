# MPSVM oblique split (paper 1, Sections 2.2 and 3.1, Algorithms 1 and 4).
#   1. The classes at a node are grouped into two hyperclasses C_p and C_n
#      (Bhattacharyya distance), so that a two-class MPSVM can be used.
#   2. Two proximal planes are found from a generalized eigenproblem.
#   3. The two angle bisectors of those planes are scored by Gini; the better one is the split.
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import eigh

from .core import FALLBACK_TO_AXIS, SplitResult, _FallbackToAxis, split_gain


# ---------------------------------------------------------------- hyperclass grouping
# Each class j at the node is modelled as a Gaussian N(mu_j, S_j). For classes j and k,
# with Sbar = (S_j + S_k)/2 and d = mu_k - mu_j, the Bhattacharyya distance is
#     F(j,k) = (1/8) d^T Sbar^-1 d + (1/2) ( ln|Sbar| - 0.5 ln|S_j| - 0.5 ln|S_k| )
# The first term measures how far apart the means are, the second how different the
# covariances are. The most distant pair (p, q) seeds C_p and C_n; every other class c
# joins C_p if F(c,p) < F(c,q), otherwise C_n.


def class_gaussian(Xc: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    # Mean mu (k,) and covariance Sigma (k, k) of the rows Xc (m_c, k) of one class.
    # The biased 1/m_c covariance is used, and eps*I is added so that Sigma is invertible
    # even for one-row classes.
    m_c,k = Xc.shape
    mu = np.zeros(k,dtype=np.float64)
    for i in range(m_c):
        for j in range(k):
            mu[j] += Xc[i,j]
    for j in range(k):
        mu[j] /= m_c
    Sigma = np.zeros((k,k),dtype=np.float64)
    for i in range(m_c):
        d = Xc[i]-mu
        for a in range(k):
            for b in range(k):
                Sigma[a,b] += d[a] * d[b]
    Sigma/=m_c
    for i in range(k):
        Sigma[i,i] += eps
    return mu,Sigma


def bhattacharyya(mu1: np.ndarray, S1: np.ndarray, mu2: np.ndarray, S2: np.ndarray) -> float:
    # Bhattacharyya distance between N(mu1, S1) and N(mu2, S2); means (k,), covariances (k, k).
    # Sbar^-1 d is obtained with solve() instead of an explicit inverse.
    S = (S1 + S2)/2.0
    d = mu2 - mu1
    M = d @ np.linalg.solve(S,d) / 8.0
    # slogdet, not log(det): det underflows to 0 for larger k (ridge eps**k), giving -inf / nan
    logdet_S = np.linalg.slogdet(S)[1]
    logdet_S1 = np.linalg.slogdet(S1)[1]
    logdet_S2 = np.linalg.slogdet(S2)[1]
    C = 0.5 * (logdet_S - 0.5*(logdet_S1 + logdet_S2))
    return C + M

def group_classes(Xs: np.ndarray, ys: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # Returns pos (m,) bool: True for samples of hyperclass C_p, False for C_n.
    # With only two classes the smaller label is taken as C_p.
    classes = np.unique(ys)
    if(len(classes)==2):
        return ys==classes[0]
    P,Q = -1,-1
    F = -1
    # a Gaussian (mu, Sigma) is fitted to every class present at the node
    class_gaussian_list = {}
    for c in classes:
        mu,Sigma = class_gaussian(Xs[ys==c],eps)
        class_gaussian_list[c] = [mu,Sigma]
    # the pair with the largest distance is found (first pair kept on ties)
    for p in classes:
        mu1,Sigma1 = class_gaussian_list[p]
        for q in classes:
            if(p==q):
                continue
            mu2, Sigma2 = class_gaussian_list[q]
            f = bhattacharyya(mu1,Sigma1,mu2,Sigma2)
            if(f>F):
                P,Q = p,q
                F = f
    # every class is assigned to the closer of the two seed classes
    Flist = {}
    muP,SigmaP = class_gaussian_list[P]
    muQ,SigmaQ = class_gaussian_list[Q]
    for c in classes:
        muC,SigmaC = class_gaussian_list[c]
        bP = bhattacharyya(muP,SigmaP,muC,SigmaC)
        bQ = bhattacharyya(muQ,SigmaQ,muC,SigmaC)
        Flist[c] = (bP<bQ)
    members = [P] + [c for c in classes if c not in (P,Q) and Flist[c]]
    res = [True]*(len(ys))
    for i in range(len(ys)):
        if(ys[i] not in members):
            res[i]=False
    return np.array(res,dtype=bool)


# ---------------------------------------------------------------- MPSVM planes
# A plane is  w.x - b = 0, and the unknowns are stacked as z = [w, b] of length k+1.
# With A = [X1, -e] (rows of C_p) and B = [X2, -e] (rows of C_n):
#     P = A^T A,  Q = B^T B,  both (k+1, k+1)
#     z^T P z = sum over C_p of (w.x - b)^2,   z^T Q z = the same sum over C_n
# Plane 1 (close to C_p, far from C_n) minimises z^T P z / z^T Q z, i.e. it is the
# eigenvector of  P z = lambda Q z  with the SMALLEST lambda. Plane 2 is the eigenvector with
# the LARGEST lambda of the same problem, so one eigh() call gives both planes.
# Small nodes make P or Q singular. Three regularizations are provided:
#   'T' : P + delta*I and Q + delta*I are used (Tikhonov)
#   'P' : if P or Q is singular, axis-parallel splits are used from this node downwards
#   'N' : if P is singular, plane 1 is chosen inside Null(P) (it passes through all of C_p)
#         so that the spread of C_n is maximal; plane 2 is found the same way with P, Q swapped


def augment(X: np.ndarray) -> np.ndarray:
    # A column of -1 is appended: X (m, k) -> [X, -e] (m, k+1).
    m,k = X.shape
    Xa = np.zeros((m,k+1),dtype=np.float64)
    for i in range(m):
        for j in range(k+1):
            if(j!=k):
                Xa[i,j] = X[i,j]
            else:
                Xa[i,j] = -1
    return Xa

def scatter_matrices(Xs: np.ndarray, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # P = A^T A from the C_p rows and Q = B^T B from the C_n rows, each (k+1, k+1).
    X1 = Xs[pos]; X2 = Xs[~pos]
    A = augment(X1); B = augment(X2)
    P = A.T @ A; Q = B.T @ B
    return P,Q


def is_rank_deficient(M: np.ndarray, tol: float = 1e-10) -> bool:
    # A symmetric PSD matrix (d, d) is treated as singular when its smallest eigenvalue is
    # below tol times the largest one (eigvalsh returns them in ascending order).
    eigenvalues = np.linalg.eigvalsh(M)
    return eigenvalues[0] <= tol * max(eigenvalues[-1],1)

def proximal_planes(P: np.ndarray, Q: np.ndarray, delta: float) -> Tuple[np.ndarray, np.ndarray]:
    # Both planes z1, z2 (d,) with d = k+1 are taken from
    #     (P + delta*I) z = lambda (Q + delta*I) z
    # eigh(A, B) returns the eigenvalues in ascending order, so column 0 is plane 1 and the
    # last column is plane 2. delta*I is added on both sides because eigh needs B positive
    # definite (B = L L^T is factorised and  L^-1 A L^-T y = lambda y  is solved, z = L^-T y).
    d,D = P.shape
    I = np.zeros((d,d),dtype=np.float64)
    for i in range(d):
        I[i,i] = 1
    eigenvalues, eigenvectors = eigh(P+delta*I,Q+delta*I)
    return eigenvectors[:,0],eigenvectors[:,-1]
def null_space_plane(P: np.ndarray, Q: np.ndarray, tol: float = 1e-10) -> Optional[np.ndarray]:
    # Null-space regularization (Algorithm 4) for one plane z (d,).
    # P is the scatter of the class the plane passes through, Q of the class it stays far from.
    # Any z = O y with O a basis of Null(P) gives z^T P z = 0. Among those, the top
    # eigenvector y of O^T Q O (r0, r0) maximises the spread of the other class.
    # None is returned if P has full rank (no null space).
    ev,V = np.linalg.eigh(P)
    O = V[:,ev<=tol*max(ev[-1],1)] # null-space basis (d, r0): eigenvectors whose eigenvalue is ~0
    # Columns of O are orthonormal
    d,r0 = O.shape
    if(r0==0):
        return None
        # if r0==0 it means P is a full rank matrix and hence return None
    eval,evec = np.linalg.eigh(O.T @ Q @ O)
    y = evec[:,-1] # eigenvector corresponding to the largest eigenvalue of O.T * Q * O
    return O @ y

def angle_bisectors(z1: np.ndarray, z2: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    # Each plane z = [w, b] (k+1,) is normalised by ||w||, then the two bisectors are formed:
    #     (+)  w = w1_hat + w2_hat,  b = b1_hat + b2_hat
    #     (-)  w = w1_hat - w2_hat,  b = b1_hat - b2_hat
    # A degenerate input plane gives no candidates; a bisector with ||w|| ~ 0 (parallel
    # planes) is dropped. The (+) candidate is listed first.
    k = len(z1)-1
    w1 = z1[:k]; w2 = z2[:k]; b1 = z1[-1]; b2 = z2[-1]
    n1 = np.linalg.norm(w1); n2 = np.linalg.norm(w2)
    if n1 < 1e-12 or n2 < 1e-12:
        return []
    out = []
    w1_hat = w1 / np.linalg.norm(w1); w2_hat = w2 / np.linalg.norm(w2)
    b1_hat = b1 / np.linalg.norm(w1); b2_hat = b2 / np.linalg.norm(w2)
    W1 = w1_hat + w2_hat; W2 = w1_hat - w2_hat
    B1 = b1_hat + b2_hat; B2 = b1_hat - b2_hat
    if(np.linalg.norm(W1)>=1e-12):
        out.append((W1,B1))
    if(np.linalg.norm(W2)>=1e-12):
        out.append((W2,B2))
    return out



def mpsvm_split(Xs: np.ndarray, ys: np.ndarray, C: int, variant: str = "T",
                delta: float = 1e-4) -> Union[SplitResult, None, _FallbackToAxis]:
    # Full MPSVM split at one node; Xs (m, k), ys (m,).
    # Returns the better bisector as a SplitResult, None if no usable plane is found,
    # or FALLBACK_TO_AXIS for variant 'P' when P or Q is singular.
    pos = group_classes(Xs,ys)
    if(not pos.any() or pos.all()):
        return None
    P,Q = scatter_matrices(Xs,pos)
    singular = is_rank_deficient(P) or is_rank_deficient(Q)
    if(variant=='P' and singular):
        return FALLBACK_TO_AXIS
    z1, z2 = None,None
    if(variant=='N' and singular):
        z1 = null_space_plane(P,Q)
        z2 = null_space_plane(Q,P)
    # planes not given by the null-space step are taken from the Tikhonov eigenproblem
    if(z1 is None or z2 is None):
        p1,p2 = proximal_planes(P,Q,delta)
        if z1 is None:
            z1 = p1
        if z2 is None:
            z2 = p2
    # each bisector is applied to the bootstrap rows and scored by Gini gain
    m, k = Xs.shape
    best_gain, best_w, best_b = -np.inf, None, None
    for w, b in angle_bisectors(z1, z2):
        go_left = np.zeros(m, dtype=bool)
        for i in range(m):
            s = 0.0
            for j in range(k):
                s += Xs[i, j] * w[j]
            go_left[i] = (s - b <= 0)
        g = split_gain(ys, go_left, C)
        if g > best_gain:                      # strict: first candidate wins ties
            best_gain, best_w, best_b = g, w, b

    if not np.isfinite(best_gain):
        return None
    return SplitResult(w=best_w, b=best_b, gain=best_gain, kind="mpsvm")
