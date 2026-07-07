# MRI-Derived and Longitudinal Predictors of Dementia in OASIS-2

A machine learning study testing whether simple proxy features extracted directly from raw structural MRI scans add predictive value beyond the clinical and FreeSurfer variables already available in the OASIS-2 dataset. Completed as an MSc thesis in Econometrics & Data Science at Vrije Universiteit Amsterdam (2026).

## Overview

Many published OASIS-2 studies report near-perfect dementia classification, but those results are often inflated by near-diagnostic predictors (CDR) and by leaking repeated visits from the same person across train/test splits. This project builds a deliberately conservative, leakage-free pipeline and uses it to ask a focused question: do cheap, reproducible image features carry information that the standard variables miss?

Two prediction tasks are evaluated:

- **Dementia classification** — visit-level Demented vs Nondemented classification.
- **Conversion prediction** — exploratory subject-level prediction of which initially Nondemented individuals later developed dementia.

## What it does

- Extracts four families of MRI proxy features from raw scans: CSF-to-brain ratio, grey-white intensity proxy, hemispheric asymmetry, and GLCM texture (contrast, homogeneity, entropy).
- Reorients every scan to RAS+ canonical orientation and builds intensity-threshold masks with `nibabel` and `scikit-image`.
- Organises predictors into nested feature sets (clinical → +FreeSurfer → +MRI proxy → all) so each block's contribution can be isolated.
- Trains and compares four classifiers (Elastic Net LR, SVM-RBF, Random Forest, Histogram Gradient Boosting) inside subject-level cross-validation pipelines with imputation and scaling fitted on training folds only.
- Assesses robustness with repeated CV and Monte Carlo stability analyses, and interprets the final model with SHAP.

## Key findings

- The MRI proxy features add **no meaningful incremental value** once cognitive scores and FreeSurfer summaries are present (ΔAUC ≈ −0.004 between the full and Clinical+FreeSurfer specifications).
- **MMSE** is the dominant predictor; **nWBV** is the strongest imaging-related contributor.
- For conversion, **annualised change features** (rates of cognitive and structural decline) outperform static single-visit measurements, suggesting trajectories matter more than snapshots.
- Including **CDR** pushes classification to near-perfect (AUC ≈ 0.999), a concrete demonstration of the circularity that inflates other OASIS-2 results — which is why it is excluded from the primary models.

The headline is a well-characterised null result, framed as a methodological contribution: a fair test of incremental value in a setting where less careful designs report over-optimistic accuracy.

## Tech stack

Python · scikit-learn · nibabel · scikit-image · SHAP · pandas · NumPy · Matplotlib · `uv` for environment management

## Repository structure

```
thesis_clean.ipynb          # full pipeline: preprocessing, feature extraction, modelling, figures
mri_features_extracted.csv  # derived MRI proxy features per session
pyproject.toml / uv.lock    # reproducible environment
figures/                    # results and methodology figures
README.md
```

## Data

This project uses **OASIS-2** (Marcus et al., 2010). The raw MRI archives and clinical spreadsheet are **not** redistributed here due to the OASIS data-use terms — they are available directly from [oasis-brains.org](https://www.oasis-brains.org/) on request. The included `mri_features_extracted.csv` contains only the derived summary features computed in this work.

## Author

Varvara Kaka — MSc Econometrics & Data Science, VU Amsterdam
