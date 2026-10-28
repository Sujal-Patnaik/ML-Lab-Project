# Evaluation used by the experiment scripts:
#   - basic metrics (accuracy, confusion matrix), written from their definitions
#   - statistical comparison of M models over D datasets (Demsar 2006), as in paper 1
#   - the critical-difference (CD) diagram
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------- metrics

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Fraction of equal entries of two (m,) label vectors.
    m = len(y_true)
    same = 0
    for i in range(m):
        if(y_true[i]==y_pred[i]):
            same += 1
    return same / m


def confusion_matrix(a: np.ndarray, b: np.ndarray, C: int) -> np.ndarray:
    # Count matrix M (C, C): M[i, j] = number of samples with a == i and b == j
    # (a, b are encoded labels 0..C-1).
    M = np.zeros((C,C), dtype=np.int64)
    for i in range(len(a)):
        M[a[i], b[i]] += 1
    return M


# ---------------------------------------------------------------- statistical tests
# Input: a score matrix S (D, M), D datasets (rows) x M models (columns), e.g. accuracies.
#   1. ranks per dataset, 1 = best, ties get the average rank; R_j = mean rank of model j
#   2. Friedman:        chi2_F = 12D / (M(M+1)) [ sum_j R_j^2 - M(M+1)^2 / 4 ],  df = M-1
#   3. Iman-Davenport:  F_F = (D-1) chi2_F / (D(M-1) - chi2_F),  df = (M-1, (M-1)(D-1))
#   4. Nemenyi:         two models differ if |R_a - R_b| >= CD = q_alpha sqrt(M(M+1) / (6D))
# With the paper's ranks (M = 12, D = 121) this gives chi2_F = 71.06, F_F = 6.77, CD = 1.51.


def rank_scores(S: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    # Per-dataset ranks (D, M), 1 = best, tied models share the average of their positions.
    D, M = S.shape
    ranks = np.zeros((D, M), dtype=np.float64)
    for d in range(D):
        for j in range(M):
            better = 0; equal = 0
            for l in range(M):
                if(S[d,l]==S[d,j]):
                    equal += 1          # includes j itself
                elif((S[d,l]>S[d,j]) == higher_is_better):
                    better += 1
            ranks[d,j] = better + (equal + 1) / 2   # tied models share the average of their positions
    return ranks


def friedman(avg_ranks: np.ndarray, D: int) -> Tuple[float, float]:
    # Friedman statistic and p-value from the average ranks (M,) over D datasets.
    M = len(avg_ranks)
    s = 0.0
    for r in avg_ranks:
        s += r**2
    chi2 = 12*D / (M*(M+1)) * (s - M*(M+1)**2 / 4)
    return float(chi2), float(sps.chi2.sf(chi2, M - 1))


def iman_davenport(chi2: float, D: int, M: int) -> Tuple[float, float]:
    # Iman-Davenport F_F (less conservative than Friedman) and its p-value.
    F = (D - 1) * chi2 / (D * (M - 1) - chi2)
    return float(F), float(sps.f.sf(F, M - 1, (M - 1) * (D - 1)))


def nemenyi_cd(M: int, D: int, alpha: float = 0.05) -> float:
    # Nemenyi critical difference; q_alpha = studentized range quantile / sqrt(2),
    # with a very large df standing in for infinity.
    q = sps.studentized_range.ppf(1 - alpha, M, 1e6) / np.sqrt(2)
    return float(q * np.sqrt(M * (M + 1) / (6 * D)))


def significant_pairs(avg_ranks: np.ndarray, names: List[str], cd: float) -> List[Tuple[str, str, float]]:
    # All pairs (better, worse, rank difference) with difference >= cd, largest first.
    M = len(avg_ranks)
    pairs = []
    for a in range(M):
        for b in range(a+1, M):
            diff = abs(avg_ranks[a] - avg_ranks[b])
            if(diff>=cd):
                if(avg_ranks[a]<=avg_ranks[b]):
                    pairs.append((names[a], names[b], float(diff)))
                else:
                    pairs.append((names[b], names[a], float(diff)))
    return sorted(pairs, key=lambda p: -p[2])


def win_tie_loss(a: np.ndarray, b: np.ndarray, tol: float = 0.0) -> Tuple[int, int, int]:
    # (wins, ties, losses) of model a against model b over the datasets; tie if |a - b| <= tol.
    wins, ties, losses = 0, 0, 0
    for i in range(len(a)):
        if(abs(a[i] - b[i])<=tol):
            ties += 1
        elif(a[i]>b[i]):
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


def summary_table(S: np.ndarray, names: List[str], alpha: float = 0.05) -> Dict[str, object]:
    # The whole procedure is run on S (D, M): average ranks, Friedman, Iman-Davenport,
    # Nemenyi CD and the significantly different pairs.
    D, M = S.shape
    ranks = rank_scores(S)
    avg = np.zeros(M, dtype=np.float64)
    for j in range(M):
        avg[j] = ranks[:,j].sum() / D
    chi2, p_f = friedman(avg, D)
    F, p_i = iman_davenport(chi2, D, M)
    cd = nemenyi_cd(M, D, alpha)
    return {"avg_ranks": {names[j]: float(avg[j]) for j in range(M)},
            "chi2": chi2, "p_friedman": p_f, "F_F": F, "p_iman": p_i, "cd": cd,
            "significant": significant_pairs(avg, names, cd)}


# ---------------------------------------------------------------- CD diagram

def plot_cd_diagram(avg_ranks: Dict[str, float], cd: float, ax):
    # Simplified critical-difference diagram on a matplotlib Axes.
    # One marker per model at its average rank (best first); each model is joined to the
    # farthest model within cd of it (not significantly different); a bar of length cd is
    # drawn above the plot.
    names = sorted(avg_ranks, key=lambda n: avg_ranks[n])     # best (lowest rank) first
    ranks = [avg_ranks[n] for n in names]
    M = len(names)
    for i in range(M):
        ax.plot(ranks[i], i, "o", color="black")
        ax.annotate(names[i], (ranks[i], i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8)
    for i in range(M):
        far = i                                   # farthest model within cd of model i
        for j in range(i+1, M):
            if(ranks[j] - ranks[i] < cd):
                far = j
        if(far>i):
            ax.plot([ranks[i], ranks[far]], [i, far], color="tab:red", lw=3, alpha=0.5)
    lo = min(ranks)
    ax.plot([lo, lo + cd], [-1, -1], color="black", lw=2)
    ax.text(lo + cd/2, -1.3, "CD", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("average rank (1 = best)")
    ax.set_yticks([])
    ax.set_ylim(M, -2)
    return ax
