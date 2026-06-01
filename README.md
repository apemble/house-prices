# House Prices: Advanced Regression Techniques

Kaggle competition — predict the sale price of residential homes in Ames, Iowa.  
Competition page: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

---

## Results

| Model | CV RMSLE | Kaggle leaderboard |
|---|---|---|
| Lasso (α=0.0005) | 0.1404 | — |
| XGBoost | 0.1257 | ~$14,100 MAE |

RMSLE is the competition metric (lower is better). The XGBoost model beat the Lasso baseline by ~10% and was used for the final submission.

---

## Approach

### Exploratory Data Analysis

Full EDA is in `notebooks/eda.ipynb`. Key findings that drove modeling decisions:

- **Target is right-skewed** (skewness 1.88) — prices are multiplicative, not additive. Modeled `log1p(SalePrice)` throughout; predictions inverted with `expm1()` at submission time.
- **Two distinct missing-value patterns:** structural NAs (e.g. `PoolQC`, `FireplaceQu`) mean the feature is absent, not that data is missing; true missing values (`LotFrontage`, `Electrical`) need imputation.
- **Diminishing marginal returns** visible in area features (`GrLivArea`, `TotalBsmtSF`, `GarageArea`) — log-transforming these tightens the linear relationship with log-price.
- **Regime change at zero** for garage and basement area — a discrete price jump exists between having one and not, on top of the continuous within-group gradient.
- **`OverallQual`** is the single strongest predictor (r = 0.79) with a visible price floor at quality ≤ 4 (land value sets a minimum regardless of build quality).
- **Collinear pairs** confirmed: `GarageYrBlt`/`YearBuilt` (r = 0.83), `TotRmsAbvGrd`/`GrLivArea` (r = 0.83). Redundant features dropped.
- **Neighborhood** spans a ~3× price range across 25 categories — target-encoded to preserve the full ordinal gradient without high-cardinality OHE.

### Preprocessing Pipeline

Implemented in `src/preprocessing.py` as a sklearn `Pipeline` + `ColumnTransformer` — fits on training data only, so it is safe inside cross-validation with no leakage.

| Step | What it does |
|---|---|
| `StructuralNAFiller` | Fills absent-feature NAs: categorical → `"None"`, numeric → `0` |
| `TrueMissingImputer` | `LotFrontage` → per-neighborhood median; `Electrical` → mode |
| `CatchallImputer` | Fills any surviving NaN in the test set (columns with no training-set NaN) |
| `BinaryFeatureAdder` | Adds `HasGarage` and `HasBasement` binary flags |
| `LogAreaTransformer` | `log1p` of `GrLivArea`, `TotalBsmtSF`, `GarageArea` |
| `ColumnDropper` | Drops `Id`, `GarageYrBlt`, `TotRmsAbvGrd` |
| `NeighborhoodTargetEncoder` | Smoothed mean of log-target per neighborhood (smoothing factor = 10) |
| `OneHotEncoder` | All remaining categorical columns; `handle_unknown='ignore'` for test safety |

### Models

All models trained with 5-fold cross-validation; CV RMSLE matches the Kaggle leaderboard metric directly.

**Ridge** — `alpha=10`. Requires `StandardScaler` (regularization penalizes all coefficients equally, so features must be on the same scale). Useful as a sanity-check baseline and for identifying which features carry signal.

**Lasso** — `alpha=0.0005`. Same scaler requirement. The small alpha is intentional — the default of 1.0 drives nearly all coefficients to zero on this dataset. Useful for implicit feature selection.

**XGBoost** — `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`, `early_stopping_rounds=50`. No feature scaling needed. Cross-validation runs as a manual fold loop (sklearn's `cross_val_score` doesn't support `eval_set` for early stopping). A fresh preprocessor is fitted per fold to avoid leakage through `NeighborhoodTargetEncoder`. The final model uses the average `best_iteration` across folds, retrained on the full dataset.

---

## Repository structure

```
house-prices/
├── data/                          # Raw Kaggle CSVs (gitignored)
│   └── home-data-for-ml-course/
│       ├── train.csv              # 1460 rows, 81 columns
│       ├── test.csv               # 1459 rows, no SalePrice
│       └── data_description.txt
├── notebooks/
│   └── eda.ipynb                  # Full exploratory data analysis
├── src/
│   ├── preprocessing.py           # sklearn Pipeline: imputation, encoding, feature engineering
│   └── train.py                   # CV training + submission generation for all three models
├── submissions/                   # Generated CSVs (gitignored)
└── CLAUDE.md                      # Project notes and conventions
```

To reproduce:

```bash
python3 src/train.py
```

Trains Ridge, Lasso, and XGBoost; prints CV scores; writes a submission CSV to `submissions/`.

---

## What I'd do next

**Feature engineering**
- Test `TotalFootprintSF = max(TotalBsmtSF, 1stFlrSF)` as an alternative to keeping both columns separately — cleaner representation of a house's total ground-floor footprint.
- Add age features: `HouseAge = YrSold - YearBuilt`, `YearsSinceRemodel = YrSold - YearRemodAdd`.
- Interaction terms: `OverallQual × GrLivArea` captures the fact that quality multiplies area value, not just adds to it.

**Outlier handling**
- Rows 524 and 1299 are partial sales of extremely large houses at well-below-market prices. Test empirically whether dropping them improves or hurts CV score — they may be anchoring predictions for large houses at the wrong price.

**Model tuning**
- `RidgeCV` / `LassoCV` for automatic alpha selection across a log-spaced grid.
- XGBoost hyperparameter search with Optuna: `max_depth`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`.
- LightGBM — often faster and slightly more accurate than XGBoost on structured tabular data.

**Ensembling**
- A weighted average of Lasso (captures linear structure) and XGBoost (captures non-linear interactions) typically outperforms either model alone on this dataset.
- Stacking with a Ridge meta-learner is a straightforward next step.

**Encoding**
- The `NeighborhoodTargetEncoder` smoothing factor (currently 10) is a hyperparameter worth tuning — lower values trust small neighborhoods more, higher values shrink toward the global mean.
- Ordinal encoding for quality columns (`ExterQual`, `KitchenQual`, `BsmtQual`, etc.) rather than OHE — these have a natural order that OHE discards.
