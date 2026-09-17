"""
Phase 3: Machine Learning Model Development

Reusable pipeline/CV/evaluation code for predicting Airbnb Rome nightly price
from the Phase 2 feature-engineered dataset. Imported by the Phase 3 notebook,
but also runnable standalone for a quick benchmark from the command line.

Usage:
    python src/model_training.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INPUT_PATH = "data/processed/listings_features.parquet"

# The column whose values define the CV groups (one neighbourhood never spans
# both the train and validation side of a fold -> prevents spatial leakage).
GROUP_COL = "neighbourhood_cleansed"
TARGET_COL = "price"

NUMERIC_FEATURES = [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "latitude",
    "longitude",
    "amenities_count",
    "distance_to_colosseum_km",
    "distance_to_vatican_km",
    "distance_to_termini_station_km",
    "distance_to_pantheon_km",
    "distance_to_nearest_hub_km",
    "host_listings_count",
    "calculated_host_listings_count",
    "number_of_reviews",
    "reviews_per_month",
    "review_scores_rating",
    "annual_availability_days",
    "occupancy_rate_365",
]

BOOLEAN_FEATURES = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "is_multi_listing_operator",
    "has_wifi",
    "has_air_conditioning",
    "has_elevator",
    "has_balcony",
    "has_parking",
    "has_pool",
    "has_heating",
    "has_kitchen",
    "has_tv",
    "has_dishwasher",
    "has_washer",
    "has_gym",
    "has_hot_tub",
    "has_dedicated_workspace",
    "has_self_check_in",
    "has_pets_allowed",
    "has_private_entrance",
    "has_waterfront",
]

CATEGORICAL_FEATURES = [
    "room_type",
    "property_type",
    "neighbourhood_cleansed",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


def load_model_frame(path: str = INPUT_PATH) -> pd.DataFrame:
    """Load the Phase 2 output and restrict to the columns Phase 3 needs."""
    df = pd.read_parquet(path)
    keep = FEATURE_COLUMNS + [TARGET_COL]
    df = df[keep].copy()
    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(float)  # nullable boolean -> float (NaN-safe)
    print(f"Model frame: {len(df):,} rows x {len(FEATURE_COLUMNS)} features.")
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Shared preprocessing for every model in the benchmark:
      - numeric: median-impute, then scale (harmless for trees, required for Ridge)
      - boolean: median-impute (missing host flags treated as "unknown" -> 0.5)
      - categorical: one-hot, bucketing rare categories (e.g. property_type
        has ~60 levels) into an "infrequent" bucket
    """
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    boolean_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30),
            )
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("bool", boolean_pipe, BOOLEAN_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def get_models() -> dict:
    """Ridge baseline plus tree-based ensemble benchmarks."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return {
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "LightGBM": LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=600,
            learning_rate=0.03,
            depth=6,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Percentage Error, as a percentage."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.median(np.abs(y_true - y_pred) / y_true) * 100)


def dollar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, MdAPE computed in real (non-log) dollar terms."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = y_true - y_pred
    return {
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAE": float(np.mean(np.abs(errors))),
        "MdAPE": mdape(y_true, y_pred),
    }


@dataclass
class CVResult:
    model_name: str
    fold_metrics: list = field(default_factory=list)  # list[dict] per fold
    oof_true: np.ndarray = None
    oof_pred: np.ndarray = None

    def summary(self) -> dict:
        df = pd.DataFrame(self.fold_metrics)
        return {
            "model": self.model_name,
            "RMSE_mean": df["RMSE"].mean(),
            "RMSE_std": df["RMSE"].std(),
            "MAE_mean": df["MAE"].mean(),
            "MAE_std": df["MAE"].std(),
            "MdAPE_mean": df["MdAPE"].mean(),
            "MdAPE_std": df["MdAPE"].std(),
        }


def spatial_group_kfold_cv(
    model, df: pd.DataFrame, n_splits: int = 5, verbose: bool = True, model_name: str | None = None
) -> CVResult:
    """
    Fit `model` (an unfitted sklearn-compatible regressor) inside a fresh
    preprocessing pipeline, evaluated with GroupKFold(n_splits) grouped by
    neighbourhood so no neighbourhood's listings appear on both sides of a
    split. Target is trained in log1p(price) space; metrics are reported
    after inverse-transforming back to real dollars.
    """
    X = df[FEATURE_COLUMNS]
    y_dollars = df[TARGET_COL].to_numpy()
    y_log = np.log1p(y_dollars)
    groups = df[GROUP_COL].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    result = CVResult(model_name=model_name or type(model).__name__)
    oof_pred = np.full(len(df), np.nan)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        pipe.fit(X.iloc[train_idx], y_log[train_idx])

        pred_log = pipe.predict(X.iloc[val_idx])
        pred_dollars = np.clip(np.expm1(pred_log), 0, None)
        oof_pred[val_idx] = pred_dollars

        metrics = dollar_metrics(y_dollars[val_idx], pred_dollars)
        result.fold_metrics.append(metrics)
        if verbose:
            val_groups = sorted(set(groups[val_idx]))
            print(
                f"  fold {fold + 1}/{n_splits} (n={len(val_idx):>5}, "
                f"{len(val_groups)} neighbourhoods held out): "
                f"RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  MdAPE={metrics['MdAPE']:.1f}%"
            )

    result.oof_true = y_dollars
    result.oof_pred = oof_pred
    return result


def run_benchmark(df: pd.DataFrame | None = None, n_splits: int = 5) -> tuple[pd.DataFrame, dict]:
    """Run every model in get_models() through spatial_group_kfold_cv and
    return (summary_table, {model_name: CVResult})."""
    if df is None:
        df = load_model_frame()

    results = {}
    for name, model in get_models().items():
        print(f"\n{name}")
        results[name] = spatial_group_kfold_cv(model, df, n_splits=n_splits, model_name=name)

    summary = pd.DataFrame([r.summary() for r in results.values()]).set_index("model")
    return summary, results


# ---------------------------------------------------------------------------
# Phase 4: native-categorical LightGBM (used for SHAP + deployment).
#
# The Phase 3 benchmark one-hot encodes categoricals inside a sklearn
# Pipeline, which is fine for comparing models but explodes property_type's
# ~60 levels into that many dummy SHAP features. For interpretability (and
# for the model we actually ship) we instead give LightGBM its native pandas
# `category` dtype support: one SHAP feature per engineered column, and NaNs
# are left for LightGBM to split on directly instead of being imputed.
# ---------------------------------------------------------------------------

LGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
)

MODEL_PATH = "models/lightgbm_price_model.joblib"


def prepare_native_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build (X, y_log, y_dollars) with categoricals as pandas 'category' dtype."""
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    y_dollars = df[TARGET_COL].to_numpy()
    y_log = np.log1p(y_dollars)
    return X, y_log, y_dollars


def fit_lightgbm_native(X: pd.DataFrame, y_log: np.ndarray, **param_overrides):
    """Fit a single LGBMRegressor directly on `X` (native categoricals, raw NaNs)."""
    from lightgbm import LGBMRegressor

    params = {**LGBM_PARAMS, **param_overrides}
    model = LGBMRegressor(**params)
    model.fit(X, y_log, categorical_feature=CATEGORICAL_FEATURES)
    return model


def spatial_oof_predictions_native(
    df: pd.DataFrame, n_splits: int = 5, verbose: bool = True, **param_overrides
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same spatial GroupKFold protocol as spatial_group_kfold_cv, but using the
    native-categorical LightGBM model so the out-of-fold predictions are
    directly comparable to (and diagnose failure modes of) the exact model
    architecture used for SHAP and deployment. Returns (y_dollars, oof_pred).
    """
    X, y_log, y_dollars = prepare_native_features(df)
    groups = df[GROUP_COL].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.full(len(df), np.nan)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
        model = fit_lightgbm_native(X.iloc[train_idx], y_log[train_idx], **param_overrides)
        pred_log = model.predict(X.iloc[val_idx])
        pred_dollars = np.clip(np.expm1(pred_log), 0, None)
        oof_pred[val_idx] = pred_dollars
        if verbose:
            metrics = dollar_metrics(y_dollars[val_idx], pred_dollars)
            print(
                f"  fold {fold + 1}/{n_splits} (n={len(val_idx):>5}): "
                f"RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  MdAPE={metrics['MdAPE']:.1f}%"
            )

    return y_dollars, oof_pred


def save_model(model, path: str = MODEL_PATH) -> None:
    """Persist a fitted model (e.g. the Phase 4 final LightGBM) for reuse in later phases."""
    import joblib
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def load_model(path: str = MODEL_PATH):
    """Load a model previously saved with save_model(). Build inputs for it with
    prepare_native_features() so categorical dtypes match what it was trained on."""
    import joblib

    return joblib.load(path)


if __name__ == "__main__":
    frame = load_model_frame()
    summary_table, _ = run_benchmark(frame)
    print("\n" + "=" * 80)
    print("Spatial 5-fold CV summary (real dollar terms)")
    print("=" * 80)
    print(summary_table.round(2).to_string())
