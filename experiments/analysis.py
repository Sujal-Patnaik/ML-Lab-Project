# Paper-style comparison from an accuracy CSV (one row per dataset x model):
#     python experiments/analysis.py --csv results/accuracy_uci.csv --outdir results/uci_analysis
# Outputs: average ranks, Friedman / Iman-Davenport / Nemenyi results (summary.json),
# the pairwise win-tie-loss table (win_tie_loss.csv) and the CD diagram (cd_diagram.pdf).
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from svf.evaluation import summary_table, win_tie_loss  # noqa: E402


def score_matrix(df: "pd.DataFrame", metric: str = "accuracy") -> Tuple[np.ndarray, List[str], List[str]]:
    # The long CSV is pivoted into S (D, M). Only datasets on which every model has a result
    # are kept, because the Friedman test needs complete rows.
    table = df.pivot_table(index="dataset", columns="model", values=metric).dropna()
    return table.to_numpy(dtype=np.float64), list(table.index), list(table.columns)


def wtl_table(S: np.ndarray, models: List[str]) -> "pd.DataFrame":
    # (M, M) table whose cell [a, b] is 'wins-ties-losses' of model a against model b.
    M = len(models)
    cells = [["" for _ in range(M)] for _ in range(M)]
    for a in range(M):
        for b in range(M):
            if(a!=b):
                w, t, l = win_tie_loss(S[:,a], S[:,b])
                cells[a][b] = f"{w}-{t}-{l}"
    return pd.DataFrame(cells, index=models, columns=models)


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description="Average ranks, Friedman / Nemenyi tests and CD diagram.")
    ap.add_argument("--csv", default="results/accuracy.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--metric", default="accuracy")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    S, datasets, models = score_matrix(df, args.metric)
    D, M = S.shape
    if(D<2 or M<2):
        raise SystemExit(f"need >= 2 complete datasets and >= 2 models, got D={D}, M={M}")
    out = summary_table(S, models)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"{D} datasets x {M} models ({args.metric})\n")
    print(pd.DataFrame(S, index=datasets, columns=models).round(4).to_string(), "\n")
    print("average ranks (1 = best):")
    for name in sorted(out["avg_ranks"], key=lambda n: out["avg_ranks"][n]):
        print(f"  {name:14s} {out['avg_ranks'][name]:.3f}")
    print(f"\nFriedman chi2 = {out['chi2']:.4f} (p = {out['p_friedman']:.4g})")
    print(f"Iman-Davenport F_F = {out['F_F']:.4f} (p = {out['p_iman']:.4g})")
    print(f"Nemenyi CD = {out['cd']:.4f}; significant pairs: {out['significant'] or 'none'}")

    summary = dict(out)
    summary["datasets"], summary["models"] = datasets, models
    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    wtl_table(S, models).to_csv(os.path.join(args.outdir, "win_tie_loss.csv"))

    # the CD diagram is drawn without a display (Agg backend)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from svf.evaluation import plot_cd_diagram
    fig, ax = plt.subplots(figsize=(7, 0.4 * M + 1.5))
    plot_cd_diagram(out["avg_ranks"], out["cd"], ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "cd_diagram.pdf"))
    print(f"\nwrote summary.json, win_tie_loss.csv, cd_diagram.pdf to {args.outdir}")


if __name__ == "__main__":
    main()
