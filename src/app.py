"""
Phase 5: Interactive Plotly Explorer & Dashboard

Streamlit app with:
  - A dual-layer map: a Municipi choropleth (median price or mean absolute
    error) under a listing scatter layer colored by room type
  - A scenario predictor: pick a neighbourhood / room type / size /
    amenities and see the winning LightGBM model's predicted price against
    the empirical distribution of comparable real listings

Usage:
    streamlit run src/app.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import model_training as mt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("airbnb_rome.app")

LISTINGS_PATH = PROJECT_ROOT / "data/processed/listings_features.parquet"
GEOJSON_PATH = PROJECT_ROOT / "data/raw/neighbourhoods.geojson"
OOF_PATH = PROJECT_ROOT / "data/processed/oof_predictions.parquet"
MODEL_PATH = PROJECT_ROOT / "models/lightgbm_price_model.joblib"

# Colorblind-safe categorical palette (Okabe-Ito) -- same one used throughout
# the Phase 3/4 notebooks, kept fixed-order for room_type so a color always
# means the same thing across every chart in this project. .streamlit/config.toml's
# primaryColor is a lightened variant of PALETTE[0] for contrast on the dark theme.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
ROME_CENTER = {"lat": 41.9028, "lon": 12.4964}
MAX_MAP_POINTS = 6000

AMENITY_LABELS = {
    "has_wifi": "Wifi",
    "has_air_conditioning": "Air conditioning",
    "has_elevator": "Elevator",
    "has_balcony": "Balcony",
    "has_parking": "Parking",
    "has_pool": "Pool",
    "has_heating": "Heating",
    "has_kitchen": "Kitchen",
    "has_tv": "TV",
    "has_dishwasher": "Dishwasher",
    "has_washer": "Washer",
    "has_gym": "Gym",
    "has_hot_tub": "Hot tub",
    "has_dedicated_workspace": "Dedicated workspace",
    "has_self_check_in": "Self check-in",
    "has_pets_allowed": "Pets allowed",
    "has_private_entrance": "Private entrance",
    "has_waterfront": "Waterfront",
}


# ---------------------------------------------------------------------------
# Startup validation & data loading
# ---------------------------------------------------------------------------


def _fail(message: str) -> None:
    """Show a user-facing error and halt the script. st.stop() raises inside a
    real Streamlit session (its own control-flow exception, unreachable past
    this point), but the explicit raise afterwards is defense-in-depth so a
    bare/test execution context can never fall through to use an unbound
    value instead of silently doing the wrong thing."""
    st.error(message)
    st.stop()
    raise RuntimeError(message)


def require_file(path: Path, hint: str) -> None:
    """Fail fast with an actionable, user-facing message instead of a raw
    traceback when a pipeline artifact hasn't been generated yet."""
    if not path.exists():
        logger.error("Missing required file: %s", path)
        _fail(f"**Missing required file:** `{path.relative_to(PROJECT_ROOT)}`\n\n{hint}")


@st.cache_data(show_spinner="Loading listings...")
def load_listings() -> pd.DataFrame:
    return pd.read_parquet(LISTINGS_PATH)


@st.cache_data(show_spinner="Loading neighbourhood boundaries...")
def load_geojson() -> dict:
    return json.loads(GEOJSON_PATH.read_text())


@st.cache_data(show_spinner="Loading out-of-fold predictions...")
def load_oof() -> pd.DataFrame:
    return pd.read_parquet(OOF_PATH)


@st.cache_resource(show_spinner="Loading price model...")
def load_model():
    return mt.load_model(str(MODEL_PATH))


def load_app_data():
    """Load every artifact, turning a corrupt/incompatible file into a clear
    st.error + stop instead of an unhandled exception mid-render."""
    try:
        listings = load_listings()
        logger.info("Loaded %d listings from %s", len(listings), LISTINGS_PATH)
    except Exception:
        logger.exception("Failed to load listings parquet")
        _fail(
            f"Could not read `{LISTINGS_PATH.name}` -- it may be corrupt or in an unexpected format. "
            "Try regenerating it with `python src/main.py`."
        )

    try:
        geojson = load_geojson()
    except Exception:
        logger.exception("Failed to load neighbourhoods geojson")
        _fail(f"Could not read `{GEOJSON_PATH.name}`.")

    try:
        oof = load_oof()
        logger.info("Loaded %d out-of-fold predictions from %s", len(oof), OOF_PATH)
    except Exception:
        logger.exception("Failed to load OOF predictions")
        _fail(
            f"Could not read `{OOF_PATH.name}` -- it may be corrupt. "
            "Regenerate it via `notebooks/interpretability.ipynb`."
        )

    try:
        model = load_model()
        logger.info("Loaded model from %s", MODEL_PATH)
    except Exception:
        logger.exception("Failed to load model artifact")
        _fail(
            f"Could not load `{MODEL_PATH.name}` -- it may be corrupt or built with an incompatible "
            "LightGBM version. Regenerate it via `notebooks/interpretability.ipynb`."
        )

    return listings, geojson, oof, model


# ---------------------------------------------------------------------------
# Scenario predictor logic
# ---------------------------------------------------------------------------


def get_profile(df: pd.DataFrame, neighbourhood: str, room_type: str) -> pd.DataFrame:
    """The closest non-trivial group of comparable real listings, used to
    fill in every feature the scenario UI doesn't expose directly (host
    attributes, review scores, distances, ...) with realistic, local values
    instead of dataset-wide defaults."""
    subset = df[(df["neighbourhood_cleansed"] == neighbourhood) & (df["room_type"] == room_type)]
    if len(subset) < 5:
        subset = df[df["neighbourhood_cleansed"] == neighbourhood]
    if len(subset) < 5:
        subset = df
    return subset


def build_scenario_row(
    profile: pd.DataFrame,
    neighbourhood: str,
    room_type: str,
    accommodates: int,
    amenities_selected: set[str],
) -> pd.DataFrame:
    row = {col: profile[col].median() for col in mt.NUMERIC_FEATURES}
    row.update({col: (1.0 if col in amenities_selected else 0.0) for col in mt.BOOLEAN_FEATURES})
    row["accommodates"] = accommodates
    row["neighbourhood_cleansed"] = neighbourhood
    row["room_type"] = room_type
    row["property_type"] = profile["property_type"].mode().iat[0]

    scenario_df = pd.DataFrame([row])[mt.FEATURE_COLUMNS]
    for col in mt.CATEGORICAL_FEATURES:
        scenario_df[col] = scenario_df[col].astype("category")
    return scenario_df


def simulate_price_distribution(
    point_prediction: float, oof: pd.DataFrame, neighbourhood: str, room_type: str, n_draws: int = 2000
) -> np.ndarray:
    """
    LightGBM only gives a point estimate, so we turn it into a distribution
    by bootstrapping *multiplicative* out-of-fold residual ratios
    (actual / predicted) from the closest matching real segment and applying
    them to the point prediction. Multiplicative (not additive) because
    Phase 4's error diagnosis showed error scales with price -- a fixed
    dollar-amount residual from a cheap listing would understate the spread
    for an expensive scenario, and vice versa.
    """
    segment = oof[(oof["neighbourhood_cleansed"] == neighbourhood) & (oof["room_type"] == room_type)]
    if len(segment) < 20:
        segment = oof[oof["neighbourhood_cleansed"] == neighbourhood]
    if len(segment) < 20:
        segment = oof

    ratios = segment["residual_ratio"].dropna().to_numpy()
    if len(ratios) == 0:
        ratios = oof["residual_ratio"].dropna().to_numpy()
    if len(ratios) == 0:
        # Degenerate: no valid ratios anywhere in the dataset. Fall back to a
        # point mass at 1.0 so the app still renders a (zero-width) result
        # instead of crashing on np.random.choice([]).
        logger.warning("No valid residual_ratio values found; falling back to a point mass at 1.0.")
        ratios = np.array([1.0])

    sampled_ratios = np.random.default_rng(0).choice(ratios, size=n_draws, replace=True)
    return np.clip(point_prediction * sampled_ratios, 0, None)


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------


def build_map_figure(
    listings: pd.DataFrame,
    oof: pd.DataFrame,
    geojson: dict,
    shading: str,
    selected_room_types: list[str],
    price_range: tuple[int, int],
) -> tuple[go.Figure, int, int]:
    if shading == "Typical price (€)":
        agg = listings.groupby("neighbourhood_cleansed", observed=True)["price"].median()
        colorscale, colorbar_title = "Blues", "Typical price (€)"
    else:
        agg = (
            oof.assign(abs_error=lambda d: d["residual"].abs())
            .groupby("neighbourhood_cleansed", observed=True)["abs_error"]
            .mean()
        )
        colorscale, colorbar_title = "Oranges", "How far off our estimates are (€)"

    fig = go.Figure()
    fig.add_trace(
        go.Choroplethmap(
            geojson=geojson,
            locations=agg.index.tolist(),
            z=agg.values.tolist(),
            featureidkey="properties.neighbourhood",
            colorscale=colorscale,
            marker_opacity=0.55,
            marker_line_width=1,
            marker_line_color="white",
            colorbar=dict(title=colorbar_title, len=0.75),
            hovertemplate=f"<b>%{{location}}</b><br>{colorbar_title}: %{{z:.0f}}<extra></extra>",
            name="",
        )
    )

    matching = listings[
        listings["room_type"].isin(selected_room_types) & listings["price"].between(*price_range)
    ]
    total_matching = len(matching)
    map_points = matching if total_matching <= MAX_MAP_POINTS else matching.sample(MAX_MAP_POINTS, random_state=0)

    room_type_colors = {rt: PALETTE[i % len(PALETTE)] for i, rt in enumerate(sorted(listings["room_type"].unique()))}
    for rt in selected_room_types:
        sub = map_points[map_points["room_type"] == rt]
        if sub.empty:
            continue
        hover_text = (
            "€" + sub["price"].round(0).astype(int).astype(str) + "/night<br>"
            + sub["property_type"].astype(str) + "<br>"
            + sub["accommodates"].astype(str) + " guests · "
            + sub["bedrooms"].fillna(0).astype(int).astype(str) + " bed(s) · "
            + sub["bathrooms"].fillna(0).round(1).astype(str) + " bath(s)<br>"
            + sub["neighbourhood_cleansed"].astype(str)
        )
        fig.add_trace(
            go.Scattermap(
                lat=sub["latitude"],
                lon=sub["longitude"],
                mode="markers",
                marker=dict(size=6, color=room_type_colors[rt], opacity=0.85),
                name=rt,
                text=hover_text,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        map_style="carto-darkmatter",
        map_zoom=10.5,
        map_center=ROME_CENTER,
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        legend=dict(title="Room type", orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    return fig, len(map_points), total_matching


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------


def render_sidebar(listings: pd.DataFrame, oof: pd.DataFrame) -> None:
    st.sidebar.markdown("## \U0001f3db️ Airbnb Rome Explorer")
    st.sidebar.caption(
        "Browse real Rome listings, and get a price estimate for a place based on what similar listings charge."
    )

    st.sidebar.metric("Listings in dataset", f"{len(listings):,}")
    st.sidebar.metric("Neighbourhoods (Municipi)", f"{listings['neighbourhood_cleansed'].nunique()}")

    with st.sidebar.expander("How accurate are these estimates?", expanded=False):
        metrics = mt.dollar_metrics(oof["price"].to_numpy(), oof["predicted_price"].to_numpy())
        st.caption("Based on comparing estimates against real prices for listings the model hadn't seen.")
        st.metric("Typical difference from the real price", f"€{metrics['MAE']:.0f}")
        st.metric("Typical difference, as a percent", f"{metrics['MdAPE']:.1f}%")
        st.metric("Difference on the trickiest listings", f"€{metrics['RMSE']:.0f}")

def render_market_explorer_tab(listings: pd.DataFrame, oof: pd.DataFrame, geojson: dict) -> None:
    filt_col1, filt_col2, filt_col3 = st.columns([1.1, 1.4, 1.5])
    with filt_col1:
        shading = st.radio("Color neighbourhoods by", ["Typical price (€)", "How far off our estimates are (€)"])
    with filt_col2:
        room_type_options = sorted(listings["room_type"].unique())
        selected_room_types = st.multiselect("Room types shown", room_type_options, default=room_type_options)
    with filt_col3:
        price_cap = int(listings["price"].quantile(0.99))
        price_range = st.slider("Price range shown (€)", 0, price_cap, (0, price_cap))

    matching_preview = listings[
        listings["room_type"].isin(selected_room_types) & listings["price"].between(*price_range)
    ]
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Listings shown", f"{len(matching_preview):,}")
    kpi2.metric(
        "Typical price",
        f"€{matching_preview['price'].median():.0f}" if len(matching_preview) else "—",
    )
    kpi3.metric("Neighbourhoods represented", f"{matching_preview['neighbourhood_cleansed'].nunique()}")

    fig, shown, total_matching = build_map_figure(listings, oof, geojson, shading, selected_room_types, price_range)
    st.plotly_chart(fig, width="stretch")
    if total_matching > shown:
        st.caption(f"Showing a random sample of {shown:,} of {total_matching:,} matching listings for performance.")
    else:
        st.caption(f"Showing all {shown:,} matching listings.")


def render_scenario_tab(listings: pd.DataFrame, oof: pd.DataFrame, model) -> None:
    st.caption(
        "Pick a neighbourhood, room type, and amenities to get a price estimate, based on patterns learned "
        "from thousands of real Rome listings. The chart shows how much prices for similar listings "
        "typically vary, so you can see the full picture, not just a single number."
    )

    room_type_options = sorted(listings["room_type"].unique())
    col1, col2 = st.columns(2)
    with col1:
        scenario_neighbourhood = st.selectbox("Neighbourhood", sorted(listings["neighbourhood_cleansed"].unique()))
    with col2:
        scenario_room_type = st.selectbox("Room type", room_type_options)

    profile = get_profile(listings, scenario_neighbourhood, scenario_room_type)
    default_accommodates = int(profile["accommodates"].median())
    default_amenities = {col for col in AMENITY_LABELS if profile[col].mean() >= 0.5}

    scenario_accommodates = st.slider("Accommodates", 1, 16, default_accommodates)
    amenity_choices = st.multiselect(
        "Amenities",
        options=list(AMENITY_LABELS.keys()),
        default=sorted(default_amenities),
        format_func=lambda k: AMENITY_LABELS[k],
    )

    scenario_df = build_scenario_row(
        profile, scenario_neighbourhood, scenario_room_type, scenario_accommodates, set(amenity_choices)
    )
    try:
        point_prediction = float(np.expm1(model.predict(scenario_df)[0]))
    except Exception:
        logger.exception(
            "Prediction failed for scenario neighbourhood=%s room_type=%s",
            scenario_neighbourhood,
            scenario_room_type,
        )
        _fail("Could not generate a prediction for this scenario -- the input didn't match the model's expected schema.")

    simulated_prices = simulate_price_distribution(point_prediction, oof, scenario_neighbourhood, scenario_room_type)

    st.divider()
    pred_col, chart_col = st.columns([1, 2])
    with pred_col:
        st.metric("Estimated price", f"€{point_prediction:.0f}/night")
        st.metric(
            "Typical price range for similar listings",
            f"€{np.percentile(simulated_prices, 5):.0f} – €{np.percentile(simulated_prices, 95):.0f}",
        )
        n_comparable = len(
            listings[
                (listings["neighbourhood_cleansed"] == scenario_neighbourhood)
                & (listings["room_type"] == scenario_room_type)
            ]
        )
        st.caption(f"{n_comparable:,} comparable real listings in this neighbourhood + room type.")

    with chart_col:
        hist_fig = go.Figure()
        hist_fig.add_trace(
            go.Histogram(
                x=simulated_prices, nbinsx=40, marker_color=PALETTE[0], opacity=0.85, name="Simulated price"
            )
        )
        hist_fig.add_vline(
            x=point_prediction, line_width=2, line_color=PALETTE[3],
            annotation_text="our estimate", annotation_position="top",
        )
        hist_fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Nightly price (€)",
            yaxis_title="How often",
            showlegend=False,
        )
        st.plotly_chart(hist_fig, width="stretch")


def render_about_tab(listings: pd.DataFrame) -> None:
    st.markdown(
        """
### Where the data comes from

The listings shown here come from [Inside Airbnb](http://insideairbnb.com/), an independent,
non-commercial project that publishes snapshots of public Airbnb listing data for cities around
the world to support research, teaching, and public discussion. It isn't operated by, or
affiliated with, Airbnb, Inc.

### What this project is

This site is a personal, educational project built to practice data cleaning, feature
engineering, and price modeling — it is **not** an official Airbnb tool, and it isn't affiliated
with or endorsed by Airbnb, Inc. or Inside Airbnb.

The price estimates on the **Scenario Predictor** tab come from a model trained on a past
snapshot of listings, and are only meant to illustrate patterns in the data (e.g. how location or
amenities relate to price). They don't reflect current market conditions and shouldn't be used to
set or evaluate real prices, book a stay, or make any financial decision.
"""
    )
    st.caption(f"This snapshot covers {len(listings):,} listings across Rome.")


# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Airbnb Rome Price Explorer", page_icon="\U0001f3db️", layout="wide")

    require_file(LISTINGS_PATH, "Run the Phase 1/2 pipeline: `python src/main.py`.")
    require_file(GEOJSON_PATH, "This ships with the repo under `data/raw/` -- check it wasn't moved or deleted.")
    require_file(OOF_PATH, "Run `notebooks/interpretability.ipynb` (Phase 4) to export out-of-fold predictions.")
    require_file(MODEL_PATH, "Run `notebooks/interpretability.ipynb` (Phase 4) to train and save the LightGBM model.")

    listings, geojson, oof, model = load_app_data()

    st.title("Airbnb Rome — Price Explorer & Scenario Predictor")
    render_sidebar(listings, oof)

    explorer_tab, scenario_tab, about_tab = st.tabs(
        ["\U0001f5fa️ Market Explorer", "\U0001f52e Scenario Predictor", "ℹ️ About & Data"]
    )
    with explorer_tab:
        render_market_explorer_tab(listings, oof, geojson)
    with scenario_tab:
        render_scenario_tab(listings, oof, model)
    with about_tab:
        render_about_tab(listings)


if __name__ == "__main__":
    main()
