# Support Vector Forest

Oblique and rotation double random forests, combined with a leakage-free PCA / SMOTE-ENN /
seed-averaging pipeline. Everything is written from scratch in NumPy / SciPy; scikit-learn is
used only for the scaler, PCA, nearest neighbours and the estimator API.

- **Paper 1:** Ganaie, Tanveer, Suganthan, Snasel, *Oblique and rotation double random forest*,
  Neural Networks 153 (2022) 496-517. This gives the 12 forest models (RaF / DRaF families with
  axis, MPSVM-T/P/N, PCA and LDA splits).
- **Paper 2:** Han & Joe, *Enhancing Machine Learning Models Through PCA, SMOTE-ENN, and
  Stochastic Weighted Averaging*, Applied Sciences 14 (2024) 9772. This gives the preprocessing,
  resampling and seed-averaging ideas.
- **Ours:** `SVF-Adaptive`, which tries the axis, LDA and MPSVM-P splits at every node and
  keeps the one with the largest Gini gain. All paper-2 steps are fitted on the training split
  only.

## Layout

```
svf/
  core.py        split / node types, Gini impurity, axis-parallel threshold search
  mpsvm.py       hyperclass grouping (Bhattacharyya distance) + MPSVM oblique split
  rotation.py    PCA / LDA rotation splits
  splitters.py   Splitter classes (axis, MPSVM, rotation, adaptive) and make_splitter
  tree.py        DoubleRandomTree: node bootstrap, feature draw, recursive growth, prediction
  forest.py      SupportVectorForest: seeds, L trees, majority vote
  pipeline.py    paper 2: scaler + global PCA, SMOTE-ENN, SVFPipeline, seed averaging
  data.py        toy and UCI dataset loaders
  evaluation.py  accuracy, confusion matrix, Friedman / Iman-Davenport / Nemenyi, CD diagram
experiments/
  configs.py           the 12 paper models + SVF-Adaptive
  run_accuracy.py      every model on every dataset -> accuracy CSV
  run_preproc_study.py baseline vs SMOTE-ENN vs global PCA (with balanced accuracy, macro F1)
  analysis.py          average ranks, statistical tests, win-tie-loss table, CD diagram
data/            MAGIC gamma telescope, page blocks, statlog shuttle (raw UCI files)
results/         all results reported in the slides
slides/          LaTeX (beamer) presentation
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
```

## Reproducing the results

All runs use 50 trees, `mtry = sqrt(n)`, seed 0 and the same split for every model
(the given train/test split for shuttle, a stratified 70/30 split otherwise).

```bash
# 1. all 13 models on the toy and the UCI datasets
python experiments/run_accuracy.py --source toy --models RaF MPRaF-T MPRaF-P MPRaF-N RaF-PCA RaF-LDA \
    DRaF MPDRaF-T MPDRaF-P MPDRaF-N DRaF-PCA DRaF-LDA SVF-Adaptive --out results/accuracy_toy.csv
python experiments/run_accuracy.py --source uci --models RaF MPRaF-T MPRaF-P MPRaF-N RaF-PCA RaF-LDA \
    DRaF MPDRaF-T MPDRaF-P MPDRaF-N DRaF-PCA DRaF-LDA SVF-Adaptive --out results/accuracy_uci.csv

# 2. statistical comparison
python experiments/analysis.py --csv results/accuracy_toy.csv --outdir results/toy_analysis
python experiments/analysis.py --csv results/accuracy_uci.csv --outdir results/uci_analysis

# 3. preprocessing study (one call per model; the parts are concatenated into preproc_study.csv)
python experiments/run_preproc_study.py --dataset page+blocks+classification --variant resample \
    --models DRaF --out results/preproc_study/parts/page+blocks+classification_resample_DRaF.csv
python experiments/run_preproc_study.py --dataset magic+gamma+telescope --variant pca \
    --models DRaF --out results/preproc_study/parts/magic+gamma+telescope_pca_DRaF.csv
```

`run_accuracy.py` skips (dataset, model) pairs already present in the output CSV, so an
interrupted run can be resumed. The UCI results were produced in parallel jobs, one per
dataset and model family (`results/uci_parts/`, logs in `results/logs/`), and then merged into
`results/accuracy_uci.csv`.

## Results (test accuracy %, UCI)

| Splitter | MAGIC RaF | MAGIC DRaF | Page RaF | Page DRaF | Shuttle RaF | Shuttle DRaF |
|---|---|---|---|---|---|---|
| Axis    | 88.19 | 88.43 | 97.38 | 96.95 | 99.993 | 99.993 |
| MPSVM-T | 87.56 | 87.73 | 97.56 | 97.44 | 99.945 | 99.959 |
| MPSVM-P | 87.87 | 87.75 | 97.38 | 97.50 | 99.993 | 99.993 |
| MPSVM-N | 87.64 | 88.17 | 97.50 | 97.50 | 99.966 | 99.966 |
| PCA     | 88.85 | 89.01 | 97.20 | 97.20 | 99.966 | 99.986 |
| LDA     | 88.40 | 88.50 | 97.44 | 97.38 | 99.972 | 99.986 |
| SVF-Adaptive | – | 88.77 | – | 97.14 | – | 99.993 |

MPDRaF-P has the best average rank on the UCI sets (5.33); SVF-Adaptive ranks first on the four
toy sets (4.50). With only 3 datasets the Friedman test is not significant (p = 0.99).

**Preprocessing study** (mean over all 13 models):

| Dataset + step | Metric | Before | After |
|---|---|---|---|
| Page blocks + SMOTE-ENN | accuracy | 97.35 | 96.31 |
| | balanced accuracy | 82.19 | **93.71** |
| | worst-class recall | 66.3 | **87.8** |
| MAGIC + global PCA (95%) | accuracy | 88.22 | 82.63 |

SMOTE-ENN trades about 1 point of accuracy for a large gain on the rare classes of the
imbalanced page-blocks data. Global PCA ignores the labels and discards low-variance
directions that still carry class information, so it lowers accuracy for every model.
