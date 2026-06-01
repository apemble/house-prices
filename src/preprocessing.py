"""
Preprocessing pipeline for the Ames house prices dataset.
Implements all decisions from the EDA Summary in notebooks/eda.ipynb.

Public API
----------
build_preprocessor() -> unfitted sklearn Pipeline
load_data()          -> (X_train, y_train, X_test)

Usage
-----
    X_train, y_train, X_test = load_data()
    pre = build_preprocessor()
    X_tr = pre.fit_transform(X_train, y_train)   # y needed for target encoding
    X_te = pre.transform(X_test)

The pipeline is safe inside sklearn cross_val_score / GridSearchCV — each fold
fits the target encoder on its own training split, preventing leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Column groups  (derived from EDA Summary, notebooks/eda.ipynb §7)
# ---------------------------------------------------------------------------

# NaN here means the feature is absent, not that data is missing.
# Fill categoricals with "None" (a valid level) and numerics with 0.
_STRUCTURAL_NA_CAT = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]
_STRUCTURAL_NA_NUM = ["MasVnrArea"]  # GarageYrBlt is dropped below

# Real data gaps — need imputation rather than "absent" fill.
_LOT_FRONTAGE = "LotFrontage"
_ELECTRICAL = "Electrical"

# Area features with diminishing marginal returns (plateauing visible in EDA scatter plots).
# log1p is safe for zero values (e.g. GarageArea=0 for houses with no garage).
_LOG_AREA_COLS = ["GrLivArea", "TotalBsmtSF", "GarageArea"]

# Binary flags to engineer — capture the discrete price jump between
# zero and non-zero area (regime change observed in EDA).
_BINARY_FLAGS = {"HasGarage": "GarageArea", "HasBasement": "TotalBsmtSF"}

# Redundant features confirmed by collinearity analysis (EDA §7).
# GarageYrBlt r=0.83 with YearBuilt; TotRmsAbvGrd r=0.83 with GrLivArea.
_DROP_COLS = ["Id", "GarageYrBlt", "TotRmsAbvGrd"]

# All object columns after custom steps, excluding Neighborhood (target-encoded separately).
_CAT_COLS = [
    "Alley", "BldgType", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "BsmtQual", "CentralAir", "Condition1", "Condition2", "Electrical", "ExterCond",
    "ExterQual", "Exterior1st", "Exterior2nd", "Fence", "FireplaceQu", "Foundation",
    "Functional", "GarageCond", "GarageFinish", "GarageQual", "GarageType", "Heating",
    "HeatingQC", "HouseStyle", "KitchenQual", "LandContour", "LandSlope", "LotConfig",
    "LotShape", "MSZoning", "MasVnrType", "MiscFeature", "PavedDrive", "PoolQC",
    "RoofMatl", "RoofStyle", "SaleCondition", "SaleType", "Street", "Utilities",
]

# All numeric columns present after custom steps (original minus dropped, plus new binary flags).
_NUM_COLS = [
    "1stFlrSF", "2ndFlrSF", "3SsnPorch", "BedroomAbvGr", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtFullBath", "BsmtHalfBath", "BsmtUnfSF", "EnclosedPorch", "Fireplaces", "FullBath",
    "GarageArea", "GarageCars", "GrLivArea", "HalfBath", "HasBasement", "HasGarage",
    "KitchenAbvGr", "LotArea", "LotFrontage", "LowQualFinSF", "MSSubClass", "MasVnrArea",
    "MiscVal", "MoSold", "OpenPorchSF", "OverallCond", "OverallQual", "PoolArea",
    "ScreenPorch", "TotalBsmtSF", "WoodDeckSF", "YearBuilt", "YearRemodAdd", "YrSold",
]

# ---------------------------------------------------------------------------
# Custom transformers — all operate on full DataFrames
# ---------------------------------------------------------------------------

class StructuralNAFiller(BaseEstimator, TransformerMixin):
    """Fill columns where NaN means 'feature is absent', not 'data is missing'.

    Categorical → "None" (treated as a valid ordinal level downstream).
    Numeric     → 0 (e.g. MasVnrArea=0 for houses with no masonry veneer).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[_STRUCTURAL_NA_CAT] = X[_STRUCTURAL_NA_CAT].fillna("None")
        X[_STRUCTURAL_NA_NUM] = X[_STRUCTURAL_NA_NUM].fillna(0)
        return X


class TrueMissingImputer(BaseEstimator, TransformerMixin):
    """Impute real data gaps.

    LotFrontage: neighborhood median (fit on train only). Lot sizes cluster by
    neighborhood, so the local median is much more accurate than the global one.
    Global median used as fallback for any neighborhood unseen at transform time.

    Electrical: global mode (only 1 missing value in training data).
    """

    def fit(self, X, y=None):
        self._nbhd_medians = X.groupby("Neighborhood")[_LOT_FRONTAGE].median()
        self._global_lot_median = X[_LOT_FRONTAGE].median()
        self._electrical_mode = X[_ELECTRICAL].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        nbhd_fill = X["Neighborhood"].map(self._nbhd_medians)
        X[_LOT_FRONTAGE] = (
            X[_LOT_FRONTAGE].fillna(nbhd_fill).fillna(self._global_lot_median)
        )
        X[_ELECTRICAL] = X[_ELECTRICAL].fillna(self._electrical_mode)
        return X


class BinaryFeatureAdder(BaseEstimator, TransformerMixin):
    """Add HasGarage and HasBasement binary flags.

    EDA showed a discrete price jump between zero and non-zero area for both
    features — a regime change that a continuous area variable alone can't model.
    The area column is kept alongside the flag to capture the within-group gradient.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for new_col, source_col in _BINARY_FLAGS.items():
            X[new_col] = (X[source_col] > 0).astype(int)
        return X


class CatchallImputer(BaseEstimator, TransformerMixin):
    """Fill any NaN that survived the specific imputation steps above.

    This handles test-set columns that had no missing values in training
    (so they weren't covered by StructuralNAFiller or TrueMissingImputer)
    but have sparse NaN in the test set — e.g. MSZoning (4), KitchenQual (1),
    Exterior1st/2nd (1), SaleType (1), GarageCars (1), GarageArea (1).

    Categorical → "None" (OHE will treat it as an unknown category and zero it out
    via handle_unknown='ignore', which is the safest fallback for 1-4 stray NaN).
    Numeric → 0 (conservative; consistent with the structural NA convention).
    """

    def fit(self, X, y=None):
        self._cat_cols = X.select_dtypes(include="object").columns.tolist()
        self._num_cols = X.select_dtypes(include="number").columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        X[self._cat_cols] = X[self._cat_cols].fillna("None")
        X[self._num_cols] = X[self._num_cols].fillna(0)
        return X


class LogAreaTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p to area features exhibiting diminishing marginal returns.

    Applied after BinaryFeatureAdder so the binary flags are based on the
    original (unlogged) values, which is more interpretable.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[_LOG_AREA_COLS] = np.log1p(X[_LOG_AREA_COLS])
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop redundant and non-predictive columns."""

    def fit(self, X, y=None):
        # Intersect with actual columns so this is safe on both train and test
        self.cols_to_drop_ = [c for c in _DROP_COLS if c in X.columns]
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop_)


class NeighborhoodTargetEncoder(BaseEstimator, TransformerMixin):
    """Replace Neighborhood with a smoothed mean of log(SalePrice) per neighborhood.

    Why target encoding over OHE: 25 neighborhoods span a ~3× price range (EDA §6).
    Target encoding preserves that full ordinal gradient in a single numeric column.
    OHE would produce 24 sparse dummy columns with noisy signals for small neighborhoods.

    Smoothing: blends the neighborhood mean toward the global mean based on sample count.
    With `smoothing=10`, a neighborhood needs ~10 samples before its mean is trusted at
    half weight. This reduces overfitting for the 4 neighborhoods with fewer than 10 rows.

    Leakage: when used inside a Pipeline, sklearn passes y (the training fold target) to
    fit(), so cross-validation folds are each encoded using only their own training split.
    """

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing

    def fit(self, X, y):
        df = pd.DataFrame({"nbhd": X["Neighborhood"], "y": np.asarray(y)})
        global_mean = df["y"].mean()
        stats = df.groupby("nbhd")["y"].agg(["mean", "count"])
        # Shrink small neighborhoods toward global mean
        weight = stats["count"] / (stats["count"] + self.smoothing)
        self._encoding = weight * stats["mean"] + (1 - weight) * global_mean
        self._global_mean = global_mean
        return self

    def transform(self, X):
        X = X.copy()
        X["Neighborhood"] = (
            X["Neighborhood"].map(self._encoding).fillna(self._global_mean)
        )
        return X


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_preprocessor() -> Pipeline:
    """Return an unfitted preprocessing Pipeline.

    Output is a dense numpy array with columns in order:
      [Neighborhood (1)] + [OHE categoricals] + [36 numerics]

    Note on scaling: numerics are passed through unscaled. Tree models (XGBoost,
    LightGBM) don't need scaling. For Ridge/Lasso, wrap the pipeline output with
    StandardScaler or add it as an additional step.
    """

    # Sequential DataFrame-level steps before column splitting
    dataframe_steps = Pipeline([
        ("structural_na",    StructuralNAFiller()),
        ("true_missing",     TrueMissingImputer()),
        ("catchall_impute",  CatchallImputer()),   # handles test-only sparse NaN
        ("binary_features",  BinaryFeatureAdder()),
        ("log_area",         LogAreaTransformer()),
        ("drop_cols",        ColumnDropper()),
    ])

    # Parallel column-level transformations
    col_transformer = ColumnTransformer(
        transformers=[
            ("neighborhood", NeighborhoodTargetEncoder(), ["Neighborhood"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _CAT_COLS),
            ("num", "passthrough", _NUM_COLS),
        ],
        remainder="drop",  # any column not listed above is discarded
    )

    return Pipeline([
        ("dataframe_steps", dataframe_steps),
        ("col_transformer",  col_transformer),
    ])


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / "data" / "home-data-for-ml-course"


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load raw CSVs and return (X_train, y_train, X_test).

    y_train is log1p(SalePrice). Models predict in log space; invert predictions
    with np.expm1() when writing submission files.
    """
    train = pd.read_csv(_DATA_DIR / "train.csv")
    test = pd.read_csv(_DATA_DIR / "test.csv")

    X_train = train.drop(columns=["SalePrice"])
    y_train = np.log1p(train["SalePrice"])
    X_test = test.copy()

    return X_train, y_train, X_test


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    pre = build_preprocessor()

    X_tr = pre.fit_transform(X_train, y_train)
    X_te = pre.transform(X_test)

    print(f"X_train processed: {X_tr.shape}")
    print(f"X_test  processed: {X_te.shape}")
    print(f"Any NaN in train:  {np.isnan(X_tr).any()}")
    print(f"Any NaN in test:   {np.isnan(X_te).any()}")
