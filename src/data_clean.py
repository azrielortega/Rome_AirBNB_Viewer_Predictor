"""
Phase 1: Ingestion, Scrubbing & Data Hygiene

Reads the raw Airbnb Rome listings export, parses currency/whitespace,
handles outliers, drops metadata columns, applies filtering rules, and
writes the cleaned subset to a compressed .parquet file.

Usage:
    python src/phase1_ingest_clean.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/listings.csv")
OUTPUT_PATH = Path("data/processed/listings_clean.parquet")

# Explicit metadata columns to drop: scrape ids, photo/profile links, host urls.
METADATA_COLUMNS = [
    "scrape_id",
    "last_scraped",
    "source",
    "calendar_last_scraped",
    "calendar_updated",
    "picture_url",
    "listing_url",
    "host_url",
    "host_profile_id",
    "host_profile_url",
    "host_thumbnail_url",
    "host_picture_url",
]

# Columns whose raw string values are boolean flags ("t"/"f").
BOOLEAN_COLUMNS = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "has_availability",
    "instant_bookable",
]

# Low-cardinality text columns worth storing as pandas categoricals for a
# smaller, faster-to-read parquet file.
CATEGORICAL_COLUMNS = [
    "room_type",
    "property_type",
    "neighbourhood_cleansed",
    "host_response_time",
]

AIRBNB_MAX_NIGHTS_CAP = 1125  # Airbnb's platform-enforced ceiling for max_nights.
MIN_NIGHTS_CAP = 365  # A minimum-stay requirement beyond a year is not realistic.


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {path}")
    return df


def parse_currency(df: pd.DataFrame) -> pd.DataFrame:
    """Convert '$120.00' style strings to float, coercing bad values to NaN."""
    df["price"] = (
        df["price"].astype("string").str.replace(r"[\$,]", "", regex=True).astype(float)
    )
    return df


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Trim leading/trailing whitespace on every string/object column."""
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        df[col] = df[col].astype("string").str.strip()
    return df


def parse_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Airbnb's 't'/'f' text flags to nullable booleans."""
    mapping = {"t": True, "f": False}
    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(mapping).astype("boolean")
    return df


def drop_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop explicit metadata columns plus any column that is entirely null."""
    present = [c for c in METADATA_COLUMNS if c in df.columns]
    df = df.drop(columns=present)
    print(f"Dropped {len(present)} metadata columns: {present}")

    fully_null = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=fully_null)
    print(f"Dropped {len(fully_null)} fully-null columns: {fully_null}")
    return df


def handle_price_outliers(df: pd.DataFrame, iqr_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Drop extreme price outliers using a wide (3x) IQR fence.

    A wider-than-standard multiplier is used deliberately: Rome listings span
    small shared rooms to entire villas/palazzos, so legitimate price spread
    is large. The 3x fence only removes pathological values (e.g. data-entry
    errors, four-figure nightly rates) rather than normal luxury listings.
    """
    valid_price = df["price"].dropna()
    q1, q3 = valid_price.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = max(0, q1 - iqr_multiplier * iqr)
    upper = q3 + iqr_multiplier * iqr

    before = len(df)
    mask_outlier = df["price"].notna() & ((df["price"] < lower) | (df["price"] > upper))
    df = df.loc[~mask_outlier].copy()
    print(
        f"Price outlier fence: [{lower:.2f}, {upper:.2f}] "
        f"(Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, x{iqr_multiplier}). "
        f"Removed {before - len(df):,} outlier rows."
    )
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Rigorous filtering rules to drop unusable/unrealistic listings."""
    before = len(df)

    df = df.dropna(subset=["price"])
    print(f"Dropped {before - len(df):,} rows with missing price.")
    before = len(df)

    df = df[df["price"] > 0]
    print(f"Dropped {before - len(df):,} rows with price = 0.")
    before = len(df)

    df = df[df["maximum_nights"] <= AIRBNB_MAX_NIGHTS_CAP]
    print(f"Dropped {before - len(df):,} rows with maximum_nights > {AIRBNB_MAX_NIGHTS_CAP}.")
    before = len(df)

    df = df[df["minimum_nights"] <= MIN_NIGHTS_CAP]
    print(f"Dropped {before - len(df):,} rows with minimum_nights > {MIN_NIGHTS_CAP}.")
    before = len(df)

    df = df[df["minimum_nights"] <= df["maximum_nights"]]
    print(f"Dropped {before - len(df):,} rows where minimum_nights > maximum_nights.")
    before = len(df)

    return df.reset_index(drop=True)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast low-cardinality text columns to category dtype for a smaller parquet file."""
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def save_parquet(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(df):,} rows x {df.shape[1]} columns to {path} ({size_mb:.1f} MB)")


def run() -> pd.DataFrame:
    df = load_data()
    df = parse_currency(df)
    df = strip_whitespace(df)
    df = parse_booleans(df)
    df = drop_metadata_columns(df)
    df = handle_price_outliers(df)
    df = apply_filters(df)
    df = optimize_dtypes(df)
    save_parquet(df)
    return df


if __name__ == "__main__":
    run()
