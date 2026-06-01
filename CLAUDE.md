# House Prices — Kaggle competition

## What this is
Kaggle "House Prices: Advanced Regression Techniques" (learn users version).
Goal: predict SalePrice for residential homes in Ames, Iowa.
Metric: RMSLE (root mean squared log error) — lower is better.
Kaggle page: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Folder structure
- data/        → raw CSVs from Kaggle, never modify these
- notebooks/   → Jupyter notebooks for exploration and analysis
- src/         → clean Python scripts (preprocessing, training, etc.)
- submissions/ → generated submission CSVs ready to upload to Kaggle

## Data files (inside data/home-data-for-ml-course/)
- train.csv             → 1460 rows, 81 columns, includes SalePrice
- test.csv              → 1459 rows, 80 columns, no SalePrice (we predict this)
- sample_submission.csv → format Kaggle expects
- data_description.txt  → full description of every column

## Stack
- Python 3.x
- pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn

## Current status
- [x] Data downloaded into data/
- [x] EDA complete — see notebooks/eda.ipynb, summary cell at end of §7
- [ ] Baseline model
- [ ] Best CV score: not yet
- [ ] Best leaderboard score: not yet

## Preferences
- Explain what you're doing before writing code
- Ask before overwriting any existing file
- Keep raw data in data/ untouched — write processed versions as new files
- Submission files named by date and score, e.g. submissions/2026-05-18_ridge_0.142.csv
