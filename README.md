# Support Vector Forest

**Oblique and rotation double random forests, combined with a leakage-free PCA / SMOTE-ENN pipeline.**

This project re-implements the twelve forest models of *Oblique and rotation double random forest*
(paper 1) from scratch, adds the data-level ideas of *Enhancing Machine Learning Models Through PCA,
SMOTE-ENN, and Stochastic Weighted Averaging* (paper 2) without their data leakage, and proposes a new
per-node adaptive splitter, **SVF-Adaptive**. Trees, splits, MPSVM, rotations, SMOTE, ENN and the
statistical tests are written in NumPy / SciPy; scikit-learn is used only for the scaler, PCA,
nearest-neighbour search and the estimator API.

*Group 3 — Sujal Patnaik, Kushagra Singhal · Mehta Family School of Data Science and Artificial
Intelligence, IIT Guwahati.*

---

## 1. Background

### Paper 1 — Ganaie, Tanveer, Suganthan, Snasel (2022), *Neural Networks* 153, 496–517

- **Random forest (RaF)** grows each tree on one bootstrap of the training set. A sample enters a
  given tree with probability $1-(1-1/N)^N \approx 0.632$, so about 37% of the data is never seen by
  that tree, and the trees stay comparatively small.
- **Double random forest (DRaF)** (Han et al., 2020) gives every tree **all N samples**. At each node
  with $N_d > 0.1N$, a fresh bootstrap of the node is drawn and used **only to choose the split**; the
  original node data is then routed to the children. Trees grow larger, which lowers bias.
- Both are **univariate** (axis-parallel) trees, which approximate an oblique boundary with many
  stair-like cuts. Paper 1 therefore proposes:
  - **Oblique DRaF (MPDRaF-T/P/N):** each split comes from a multisurface proximal SVM (MPSVM).
    Multi-class nodes are first reduced to two *hyperclasses* with the Bhattacharyya distance
    (Algorithm 1). With $P = [X_1, -e]^\top[X_1, -e]$ and $Q = [X_2, -e]^\top[X_2, -e]$, the two
    proximal planes solve $Pz = \lambda Qz$ (smallest and largest $\lambda$), and the better of their
    two angle bisectors is the split. Small nodes make $P$ or $Q$ singular, handled by **T**ikhonov
    regularization, an axis-**P**arallel fallback for the whole subtree, or a **N**ull-space solution
    (Algorithm 4).
  - **Rotation DRaF (DRaF-PCA / DRaF-LDA):** the drawn features at each node are rotated by PCA (total
    scatter) or LDA ($S_b a = \lambda S_w a$), and the best axis cut is searched in the rotated space.
- Each model has a RaF twin (one bootstrap at the root), giving **12 models**. On 121 UCI datasets
  with 50 trees and $\mathrm{mtry}=\sqrt n$, DRaF-LDA ranks best (5.04) and MPDRaF-P second (5.45);
  double forests also grow bigger trees and show lower bias and variance.

### Paper 2 — Han & Joe (2024), *Applied Sciences* 14, 9772

- A six-step pipeline on the imbalanced Titanic data (61.6% / 38.4%): feature engineering → **PCA** →
  **resampling** → hyper-parameter search → stacking / voting ensemble → "SWA".
- **ENN + SMOTE** was its best resampler (CV accuracy 0.892 vs. 0.841 for SMOTETomek). **PCA** closed
  the train–test gap (random forest: 14.4 → 0.2 points) but also lowered test accuracy.
- Two issues matter for us: the preprocessing statistics were computed on **train + test joined**, and
  resampling was applied to the data before model training, so test information leaks into
  training (the resampling appears to precede the cross-validation folds as well). Its
  "Stochastic Weighted Averaging" retrains the model with different random states and
  averages the predictions, which is **seed (ensemble) averaging**, not SWA of network weights.

## 2. What this project does

| Part | Source | Where |
|---|---|---|
| RaF / DRaF trees with node bootstrap, majority vote | paper 1 | `svf/tree.py`, `svf/forest.py` |
| Axis split, MPSVM-T/P/N split, PCA / LDA rotation split | paper 1 | `svf/core.py`, `svf/mpsvm.py`, `svf/rotation.py` |
| Hyperclass grouping by Bhattacharyya distance | paper 1, Alg. 1 | `svf/mpsvm.py` |
| Friedman, Iman-Davenport, Nemenyi CD, win-tie-loss | paper 1, Sec. 5 | `svf/evaluation.py` |
| StandardScaler, global PCA, SMOTE, ENN | paper 2 | `svf/pipeline.py` |
| **Leak-free pipeline** — every step fitted on the training split only | ours | `SVFPipeline` |
| **SVF-Adaptive** — per-node choice among axis, LDA and MPSVM-P | ours | `AdaptiveSplitter` |

**SVF-Adaptive.** At every node, the axis, LDA and MPSVM-P splitters are run on the same node
bootstrap, and the split with the largest Gini gain is kept. Paper 1 shows that no single splitter
wins everywhere; choosing per node lets each region of the data use the geometry that fits it. It is
related to the heterogeneous oblique random forest of Katuwal, Suganthan & Zhang (2020), cited in
paper 1, but works inside the double forest and also includes axis and rotation splits.

**Pipeline.**

```
train split ──> StandardScaler ──> [global PCA] ──> [SMOTE → ENN] ──> 50 double random trees ──> majority vote
test split  ──> same scaler    ──> same PCA     ─────(never resampled)──────────────────────┘
```

Every node stores a hyperplane `(feats, w, b)` and sends a sample left when `x[feats] @ w - b <= 0`;
an axis split is the special case `w = e_j`, so all twelve models share one tree implementation.

![Univariate vs. multivariate split](images/figures/split_geometry.png)

*Left: an axis-parallel tree needs 29 nodes and a staircase for a diagonal boundary. Middle: an MPSVM
tree needs 3 nodes. Right: the two MPSVM proximal planes at the root and the bisector that becomes the
split. Drawn with the project's own `DoubleRandomTree`, `AxisSplitter` and `MPSVMSplitter`.*

### Differences from the papers

- **Data and splits:** 3 UCI datasets and 4 scikit-learn toy sets instead of 121 UCI datasets;
  shuttle uses its given train/test split, the others a stratified 70/30 split (seed 0).
- **SMOTE-ENN order:** SMOTE first, then ENN (the standard SMOTEENN), with $k=5$ for SMOTE and $k=3$
  for ENN; a sample is kept only if all 3 neighbours share its label. Paper 2 applies ENN first and
  removes only majority-class samples.
- **Regularization details:** a small ridge is added to the LDA within-class scatter and to the class
  covariances of the Bhattacharyya distance, so both are always invertible.
- **Not reproduced:** paper 1's kappa-error diversity and bias–variance analyses, and paper 2's
  hyper-parameter search and stacking / voting ensemble. Seed averaging is implemented
  (`SeedAveragedClassifier`) but not used in the experiments.

## 3. Repository layout

```
svf/
  core.py        split / node types, Gini impurity, axis-parallel threshold search
  mpsvm.py       hyperclass grouping (Bhattacharyya distance) + MPSVM oblique split (T / P / N)
  rotation.py    PCA / LDA rotation splits
  splitters.py   Splitter classes (axis, MPSVM, rotation, adaptive) and make_splitter
  tree.py        DoubleRandomTree: node bootstrap, feature draw, recursive growth, prediction
  forest.py      SupportVectorForest: one seed per tree, L trees, majority vote
  pipeline.py    paper 2: scaler + global PCA, SMOTE, ENN, SVFPipeline, seed averaging
  data.py        toy and UCI dataset loaders
  evaluation.py  accuracy, confusion matrix, Friedman / Iman-Davenport / Nemenyi, CD diagram
experiments/
  configs.py           the 12 paper models + SVF-Adaptive
  run_accuracy.py      every model on every dataset -> accuracy CSV (resumable)
  run_preproc_study.py baseline vs. SMOTE-ENN vs. global PCA (balanced accuracy, macro F1, worst recall)
  analysis.py          average ranks, statistical tests, win-tie-loss table, CD diagram
  make_figures.py      the presentation figures, drawn from svf/ and results/
data/            MAGIC gamma telescope, page blocks, statlog shuttle (raw UCI files)
results/         every result reported in the presentation
main.tex         beamer presentation; slides in segments/, figures in images/
```

## 4. Setup and reproduction

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
```

All runs use 50 trees, $\mathrm{mtry}=\sqrt n$, minleaf 1, seed 0 and the same split for every model.

```bash
MODELS="RaF MPRaF-T MPRaF-P MPRaF-N RaF-PCA RaF-LDA DRaF MPDRaF-T MPDRaF-P MPDRaF-N DRaF-PCA DRaF-LDA SVF-Adaptive"

# 1. all 13 models on the toy and the UCI datasets
python experiments/run_accuracy.py --source toy --models $MODELS --out results/accuracy_toy.csv
python experiments/run_accuracy.py --source uci --models $MODELS --out results/accuracy_uci.csv

# 2. statistical comparison (ranks, Friedman / Nemenyi, win-tie-loss, CD diagram)
python experiments/analysis.py --csv results/accuracy_toy.csv --outdir results/toy_analysis
python experiments/analysis.py --csv results/accuracy_uci.csv --outdir results/uci_analysis

# 3. preprocessing study, one call per model (variants: baseline | resample | pca)
python experiments/run_preproc_study.py --dataset page+blocks+classification --variant resample \
    --models DRaF --out results/preproc_study/parts/page+blocks+classification_resample_DRaF.csv

# 4. presentation figures (PDF, plus PNG copies)
python experiments/make_figures.py --preview images/figures
```

`run_accuracy.py` skips (dataset, model) pairs already in the output CSV, so an interrupted run can be
resumed. The UCI results were produced by separate jobs per dataset and model family
(`results/uci_parts/`, logs in `results/logs/`) and merged into `results/accuracy_uci.csv`; the
preprocessing parts are merged into `results/preproc_study/preproc_study.csv`.

## 5. Results

### UCI datasets (test accuracy %)

| Splitter | MAGIC RaF | MAGIC DRaF | Page RaF | Page DRaF | Shuttle RaF | Shuttle DRaF |
|---|---|---|---|---|---|---|
| Axis    | 88.19 | 88.43 | 97.38 | 96.95 | **99.993** | **99.993** |
| MPSVM-T | 87.56 | 87.73 | **97.56** | 97.44 | 99.945 | 99.959 |
| MPSVM-P | 87.87 | 87.75 | 97.38 | 97.50 | **99.993** | **99.993** |
| MPSVM-N | 87.64 | 88.17 | 97.50 | 97.50 | 99.966 | 99.966 |
| PCA     | 88.85 | **89.01** | 97.20 | 97.20 | 99.966 | 99.986 |
| LDA     | 88.40 | 88.50 | 97.44 | 97.38 | 99.972 | 99.986 |
| SVF-Adaptive | – | 88.77 | – | 97.14 | – | **99.993** |

MAGIC: 13,314 / 5,706 samples, 10 features, 2 classes. Page blocks: 3,831 / 1,642, 10 features,
5 classes, imbalance 172:1. Shuttle: 43,500 / 14,500, 9 features, 7 classes.

![Average ranks](images/figures/ranks.png)

- **MPDRaF-P** has the best average rank (5.33), in line with paper 1, where it is 2nd (5.45).
- **DRaF-PCA** gives the best MAGIC accuracy (89.01%, +0.6 over DRaF).
- **SVF-Adaptive** ties 2nd on the UCI sets (6.00) and ranks **1st on the four toy sets** (4.50,
  ahead of DRaF and DRaF-LDA at 5.38), with 100% on iris and wine and 96.49% on breast cancer.
- With only 3 UCI datasets, the differences between models are small (within about 1.5 points) and
  the Friedman test is not significant (p = 0.99); paper 1 needed 121 datasets for significance.

### Double vs. standard forests (mean over the 6 splitters)

| | RaF family | DRaF family |
|---|---|---|
| Unique samples per tree | ≈ 63.2% | 100% |
| Mean nodes per tree — MAGIC | 3058 | **4586** (+50%) |
| Mean nodes per tree — page blocks | 240 | **345** (+44%) |
| Mean nodes per tree — shuttle | 174 | **200** (+15%) |

DRaF beats its RaF twin in 9 of the 18 dataset–splitter pairs, with 5 ties.

### Preprocessing study (mean over all 13 models)

![Preprocessing study](images/figures/preprocessing.png)

| Dataset + step | Metric | Before | After |
|---|---|---|---|
| Page blocks + SMOTE-ENN | accuracy | 97.35 | 96.31 |
| | balanced accuracy | 82.19 | **93.71** |
| | worst-class recall | 66.3 | **87.8** |
| MAGIC + global PCA (95% variance) | accuracy | 88.22 | 82.63 |

- **SMOTE-ENN helps on imbalanced data.** Class 1 is about 90% of page blocks, so Gini splits favour
  it and rare classes end up in majority leaves. Balancing the classes and removing overlapping border
  points raises balanced accuracy for all 13 models, at the cost of about 1 point of plain accuracy.
- **Global PCA hurts.** PCA ignores the labels and keeps high-variance directions; the 3 dropped
  components hold only 5% of the variance but still carry class signal. Accuracy falls for all 13
  models, and the PCA / LDA splitters lose as much as the axis splitter, so the loss comes from
  discarded information, not from the rotation. Rotations work best at the node level, with all
  dimensions kept, as in DRaF-PCA / DRaF-LDA.

## 6. Conclusions

1. Node-level bootstrapping grows bigger trees (+44–50% nodes on MAGIC and page blocks) at similar
   accuracy.
2. Oblique and rotation splits pay off: DRaF-PCA is best on MAGIC and MPDRaF-P has the best average
   rank, consistent with paper 1.
3. SMOTE-ENN is the right paper-2 step for imbalanced data when the rare classes matter; global PCA
   should not be applied before these forests.
4. Fitting all preprocessing and resampling inside the training split keeps the evaluation honest,
   unlike the train + test preprocessing of paper 2.
5. SVF-Adaptive is competitive on UCI and best on the toy sets; more datasets are needed to show a
   significant difference.

## References

1. M. A. Ganaie, M. Tanveer, P. N. Suganthan, V. Snasel. *Oblique and rotation double random forest.*
   Neural Networks 153 (2022) 496–517.
2. Y. Han, I. Joe. *Enhancing Machine Learning Models Through PCA, SMOTE-ENN, and Stochastic Weighted
   Averaging.* Applied Sciences 14 (2024) 9772.
3. S. Han, H. Kim, Y.-S. Lee. *Double random forest.* Machine Learning 109 (2020) 1569–1586.
4. R. Katuwal, P. N. Suganthan, L. Zhang. *Heterogeneous oblique random forest.* Pattern Recognition
   99 (2020) 107078.
5. J. Demšar. *Statistical comparisons of classifiers over multiple data sets.* JMLR 7 (2006) 1–30.
6. D. Dua, C. Graff. *UCI Machine Learning Repository* (2017).
