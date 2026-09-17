"""
Phase 2: Feature Engineering (Physical & Spatial)

Takes the Phase 1 cleaned listings and derives:
  - Physical spec columns (bathrooms, bedrooms, beds, accommodates)
  - Binary amenity indicator columns parsed from the raw amenities JSON
  - Geodesic distance (km) from each listing to core Rome tourist hubs
  - Official borough assignment via point-in-polygon against neighbourhoods.geojson
  - Host & market attributes (multi-listing operator flag, review velocity,
    annual availability)

Usage:
    python src/phase2_feature_engineering.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import Point, shape
from shapely.prepared import prep

INPUT_PATH = Path("data/processed/listings_clean.parquet")
GEOJSON_PATH = Path("data/raw/neighbourhoods.geojson")
OUTPUT_PATH = Path("data/processed/listings_features.parquet")

# lat, lon of core Rome tourist hubs.
TOURIST_HUBS = {
    "colosseum": (41.8902, 12.4922),
    "vatican": (41.9022, 12.4533),
    "termini_station": (41.9010, 12.5017),
    "pantheon": (41.8986, 12.4769),
}

# High-impact amenities to flag, as case-insensitive regex patterns matched
# against the joined amenities text. Word-boundaried to avoid false positives
# (e.g. "\bpool\b" must not match "Whirlpool", "\bwasher\b" must not match
# "Dishwasher").
AMENITY_PATTERNS = {
    "has_wifi": r"\bwifi\b",
    "has_air_conditioning": r"\bair conditioning\b|\bac unit\b",
    "has_elevator": r"\belevator\b",
    "has_balcony": r"\bbalcony\b",
    "has_parking": r"\bparking\b",
    "has_pool": r"\bpool\b(?!\s*(?:table|view))",
    "has_heating": r"\bheating\b",
    "has_kitchen": r"\bkitchen\b",
    "has_tv": r"\btv\b",
    "has_dishwasher": r"\bdishwasher\b",
    "has_washer": r"\bwasher\b",
    "has_gym": r"\bgym\b",
    "has_hot_tub": r"\bhot tub\b",
    "has_dedicated_workspace": r"\bdedicated workspace\b",
    "has_self_check_in": r"\bself check-in\b",
    "has_pets_allowed": r"\bpets allowed\b",
    "has_private_entrance": r"\bprivate entrance\b",
    "has_waterfront": r"\bwaterfront\b",
}
_COMPILED_AMENITY_PATTERNS = {
    col: re.compile(pattern, re.IGNORECASE) for col, pattern in AMENITY_PATTERNS.items()
}


def load_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {path}")
    return df


def _parse_bathrooms_text(text: str) -> float:
    """'1.5 baths' -> 1.5, '1 shared bath' -> 1.0, 'Half-bath' -> 0.5."""
    if pd.isna(text):
        return float("nan")
    match = re.search(r"[\d.]+", text)
    if match:
        return float(match.group())
    if "half" in text.lower():
        return 0.5
    return float("nan")


def extract_physical_specs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure bathrooms/bedrooms/beds/accommodates are clean numeric columns.

    - bathrooms: backfilled from bathrooms_text where the numeric column is
      missing (bathrooms_text has far fewer nulls).
    - bedrooms: a null value on Airbnb conventionally denotes a studio, so it
      is imputed to 0.
    - beds: a null value is imputed to 1 (every bookable listing sleeps at
      least one person on at least one bed).
    """
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms_text"].apply(_parse_bathrooms_text))
    df["bedrooms"] = df["bedrooms"].fillna(0)
    df["beds"] = df["beds"].fillna(1)
    print(
        f"Physical specs: bathrooms null={df['bathrooms'].isna().sum()}, "
        f"bedrooms null={df['bedrooms'].isna().sum()}, beds null={df['beds'].isna().sum()}, "
        f"accommodates null={df['accommodates'].isna().sum()}"
    )
    return df


def parse_amenities(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the amenities JSON array string into an amenity count plus binary flags."""
    amenity_lists = df["amenities"].apply(json.loads)
    df["amenities_count"] = amenity_lists.apply(len)

    joined = amenity_lists.apply(lambda items: " | ".join(items))
    for col, pattern in _COMPILED_AMENITY_PATTERNS.items():
        df[col] = joined.str.contains(pattern, regex=True)

    flag_cols = list(_COMPILED_AMENITY_PATTERNS)
    print("Amenity flag prevalence:")
    print((df[flag_cols].mean() * 100).round(1).sort_values(ascending=False).to_string())
    return df


def compute_hub_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Geodesic (WGS-84 ellipsoid) distance in km from each listing to each tourist hub."""
    coords = list(zip(df["latitude"], df["longitude"]))
    for hub, hub_coord in TOURIST_HUBS.items():
        col = f"distance_to_{hub}_km"
        df[col] = [round(geodesic(c, hub_coord).km, 3) for c in coords]

    hub_distance_cols = [f"distance_to_{hub}_km" for hub in TOURIST_HUBS]
    df["distance_to_nearest_hub_km"] = df[hub_distance_cols].min(axis=1)
    print("Hub distance summary (km):")
    print(df[hub_distance_cols + ["distance_to_nearest_hub_km"]].describe().round(2).to_string())
    return df


def assign_borough(df: pd.DataFrame, geojson_path: Path = GEOJSON_PATH) -> pd.DataFrame:
    """
    Point-in-polygon spatial join of each listing's lat/lon against the
    official borough (municipio) boundaries in neighbourhoods.geojson.
    """
    features = json.load(open(geojson_path))["features"]
    boroughs = [(f["properties"]["neighbourhood"], prep(shape(f["geometry"]))) for f in features]

    def locate(lat: float, lon: float) -> str | None:
        point = Point(lon, lat)
        for name, polygon in boroughs:
            if polygon.contains(point):
                return name
        return None

    df["borough"] = [locate(lat, lon) for lat, lon in zip(df["latitude"], df["longitude"])]

    unassigned = df["borough"].isna().sum()
    if unassigned:
        print(f"Warning: {unassigned} listings did not fall inside any borough polygon.")
    if "neighbourhood_cleansed" in df.columns:
        mismatches = (df["borough"] != df["neighbourhood_cleansed"]).sum()
        print(f"Borough spatial join: {unassigned} unassigned, {mismatches} disagree with neighbourhood_cleansed.")
    return df


def add_host_market_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-listing operator flag, review velocity, and annual availability."""
    df["is_multi_listing_operator"] = df["calculated_host_listings_count"] > 1
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0.0)
    df["annual_availability_days"] = df["availability_365"]
    df["occupancy_rate_365"] = 1 - (df["annual_availability_days"] / 365)

    print(
        f"Multi-listing operators: {df['is_multi_listing_operator'].mean() * 100:.1f}% of listings. "
        f"Median reviews/month: {df['reviews_per_month'].median():.2f}. "
        f"Median annual availability: {df['annual_availability_days'].median():.0f} days."
    )
    return df


def save_parquet(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(df):,} rows x {df.shape[1]} columns to {path} ({size_mb:.1f} MB)")


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_data()
    df = extract_physical_specs(df)
    df = parse_amenities(df)
    df = compute_hub_distances(df)
    df = assign_borough(df)
    df = add_host_market_attributes(df)
    save_parquet(df)
    return df


if __name__ == "__main__":
    run()
