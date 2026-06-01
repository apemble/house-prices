"""
Baseline models for the Ames house prices competition: Ridge, Lasso, XGBoost.

All models train with 5-fold cross-validation scored by RMSLE (root mean
squared log error) — the same metric Kaggle uses for this competition.
Because y is log1p(SalePrice), RMSE in log space equals RMSLE on raw prices.

Usage (from project root):
    python3 src/train.py

Output:
    submissions/YYYY-MM-DD_{model}_{rmsle}.csv
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Allow `from preprocessing import ...` when run as python3 src/train.py
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import build_preprocessor, load_data

SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
N_FOLDS = 5
RANDOM_STATE = 42

# XGBoost hyperparameters shared by the CV and final-fit models.
# early_stopping_rounds is excluded here so we can omit it cleanly for the
# final fit (which trains a fixed number of trees on the full dataset).
_XGB_BASE_PARAMS = dict(
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",    # fastest CPU method; replace with "gpu_hist" if GPU available
    random_state=RANDOM_STATE,
)
EARLY_STOPPING_ROUNDS = 50
XGB_CV_ESTIMATORS = 1000  # upper bound; early stopping finds the true optimum per fold


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def make_pipeline(model) -> Pipeline:
    """Combine preprocessor + scaler + model into one sklearn Pipeline.

    StandardScaler is required for Ridge and Lasso: L2/L1 regularisation
    penalises all coefficients equally, so every feature must be on the same
    scale. The scaler is applied after the preprocessor (OHE + target encoding
    outputs + numeric passthrough) so it sees the final numeric matrix.
    """
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("scaler",        StandardScaler()),
        ("model",         model),
    ])


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cv_rmsle(pipeline, X, y) -> tuple[float, float]:
    """Return (mean, std) RMSLE across N_FOLDS using the given pipeline.

    y is log1p(SalePrice), so RMSE in log space == RMSLE on the raw target.
    cross_val_score returns negative values; we negate to get positive RMSLE.
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline, X, y,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return float(-scores.mean()), float(scores.std())


# ---------------------------------------------------------------------------
# Submission writer
# ---------------------------------------------------------------------------

def make_submission(pipeline, X_train, y_train, X_test, model_name, cv_score) -> Path:
    """Fit pipeline on the full training set and write a Kaggle submission CSV.

    Saves to submissions/YYYY-MM-DD_{model_name}_{cv_score:.4f}.csv.
    Predictions are in log space and must be inverted with expm1().
    """
    # Grab test IDs before the pipeline drops the Id column during transform
    test_ids = X_test["Id"].values

    print(f"Fitting {model_name} on full training set...")
    pipeline.fit(X_train, y_train)

    # Invert log1p to recover dollar SalePrice values
    log_preds = pipeline.predict(X_test)
    sale_prices = np.expm1(log_preds)

    submission = pd.DataFrame({"Id": test_ids, "SalePrice": sale_prices})

    today = date.today().strftime("%Y-%m-%d")
    filename = f"{today}_{model_name}_{cv_score:.4f}.csv"
    out_path = SUBMISSIONS_DIR / filename
    submission.to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# XGBoost cross-validation (manual loop — required for early stopping)
# ---------------------------------------------------------------------------

def cv_xgb(X, y) -> tuple[float, float, int]:
    """5-fold CV for XGBoost with per-fold early stopping.

    Returns (mean_rmsle, std_rmsle, avg_best_n_estimators).

    Why not cross_val_score: early stopping requires an eval_set passed to
    fit() at the time of the call, which cross_val_score doesn't support.
    We loop manually so we can supply each fold's val split as the eval set.

    Each fold gets a fresh build_preprocessor() fitted on its training rows
    only, so NeighborhoodTargetEncoder never sees the validation targets —
    no leakage even though the encoding step requires y.
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_rmsles: list[float] = []
    fold_best_n: list[int] = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr,  y_tr  = X.iloc[tr_idx],  y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Fresh preprocessor per fold — fits on training split, transforms val
        pre = build_preprocessor()
        X_tr_proc  = pre.fit_transform(X_tr, y_tr)
        X_val_proc = pre.transform(X_val)

        model = XGBRegressor(
            n_estimators=XGB_CV_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **_XGB_BASE_PARAMS,
        )
        model.fit(
            X_tr_proc, y_tr,
            eval_set=[(X_val_proc, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val_proc)

        # RMSLE: y_val is log1p(SalePrice), so RMSE in log space == RMSLE
        fold_rmsle = float(np.sqrt(np.mean((y_val.values - preds) ** 2)))

        # best_iteration is 0-indexed; +1 gives the actual tree count used
        best_n = model.best_iteration + 1

        fold_rmsles.append(fold_rmsle)
        fold_best_n.append(best_n)
        print(f"  Fold {fold}: RMSLE={fold_rmsle:.4f}  best_n_estimators={best_n}")

    return float(np.mean(fold_rmsles)), float(np.std(fold_rmsles)), int(np.mean(fold_best_n))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    X_train, y_train, X_test = load_data()
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    # ------------------------------------------------------------------
    # Ridge baseline
    # alpha=10 is stronger than sklearn's default of 1.0 and tends to be
    # a better starting point for this dataset before tuning.
    # ------------------------------------------------------------------
    ridge_pipeline = make_pipeline(Ridge(alpha=10.0))
    print(f"\nRidge — {N_FOLDS}-fold CV...")
    ridge_mean, ridge_std = cv_rmsle(ridge_pipeline, X_train, y_train)
    print(f"  RMSLE: {ridge_mean:.4f} ± {ridge_std:.4f}")

    # ------------------------------------------------------------------
    # Lasso baseline
    # alpha=0.0005 is deliberately small — sklearn's default of 1.0 drives
    # almost all coefficients to zero on this dataset (too aggressive).
    # max_iter raised to ensure convergence at this small alpha.
    # ------------------------------------------------------------------
    lasso_pipeline = make_pipeline(Lasso(alpha=0.0005, max_iter=10_000))
    print(f"\nLasso — {N_FOLDS}-fold CV...")
    lasso_mean, lasso_std = cv_rmsle(lasso_pipeline, X_train, y_train)
    print(f"  RMSLE: {lasso_mean:.4f} ± {lasso_std:.4f}")

    # ------------------------------------------------------------------
    # XGBoost
    # Manual fold loop (see cv_xgb) supports early stopping per fold.
    # No StandardScaler — tree models are invariant to feature scale.
    # After CV we average best_n_estimators across folds and use that fixed
    # count for the final fit on the full training set (no eval_set needed).
    # ------------------------------------------------------------------
    print(f"\nXGBoost — {N_FOLDS}-fold CV "
          f"(max_estimators={XGB_CV_ESTIMATORS}, early_stopping={EARLY_STOPPING_ROUNDS})...")
    xgb_mean, xgb_std, best_n = cv_xgb(X_train, y_train)
    print(f"  RMSLE: {xgb_mean:.4f} ± {xgb_std:.4f}  (avg best_n_estimators={best_n})")

    # ------------------------------------------------------------------
    # Summary
    # RMSLE is the Kaggle leaderboard metric — these numbers map directly
    # to what you'll see after uploading a submission CSV.
    # ------------------------------------------------------------------
    print("\n--- CV Summary (RMSLE — lower is better, matches Kaggle leaderboard) ---")
    print(f"  Ridge   {ridge_mean:.4f} ± {ridge_std:.4f}")
    print(f"  Lasso   {lasso_mean:.4f} ± {lasso_std:.4f}")
    print(f"  XGBoost {xgb_mean:.4f} ± {xgb_std:.4f}  (best_n={best_n})")

    # ------------------------------------------------------------------
    # Submission
    # If XGBoost beats Lasso, generate an XGBoost submission.
    # Otherwise generate one for whichever linear model scored best.
    # ------------------------------------------------------------------
    if xgb_mean < lasso_mean:
        print(f"\nXGBoost beats Lasso ({xgb_mean:.4f} < {lasso_mean:.4f})")
        # Final XGBoost pipeline: preprocessor → XGBRegressor with fixed best_n,
        # no early_stopping_rounds (no eval_set on the full training set).
        xgb_final_pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", XGBRegressor(n_estimators=best_n, **_XGB_BASE_PARAMS)),
        ])
        make_submission(xgb_final_pipeline, X_train, y_train, X_test, "xgboost", xgb_mean)
    else:
        print(f"\nXGBoost did not beat Lasso ({xgb_mean:.4f} >= {lasso_mean:.4f})")
        if ridge_mean <= lasso_mean:
            best_name, best_pipeline, best_score = "ridge", ridge_pipeline, ridge_mean
        else:
            best_name, best_pipeline, best_score = "lasso", lasso_pipeline, lasso_mean
        print(f"Falling back to best linear model: {best_name} ({best_score:.4f})")
        make_submission(best_pipeline, X_train, y_train, X_test, best_name, best_score)


if __name__ == "__main__":
    main()
