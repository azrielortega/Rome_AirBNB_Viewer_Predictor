# Airbnb Rome — Price Analysis & Prediction

An end-to-end pipeline on Inside Airbnb's Rome listings: data cleaning,
feature engineering, spatially-cross-validated price modeling, SHAP-based
interpretability, and an interactive Streamlit dashboard.

## Pipeline overview

| Phase | What it does | Where |
|---|---|---|
| 1 | Ingestion & data hygiene: currency parsing, outlier handling, filtering rules | `src/data_clean.py` |
| 2 | Feature engineering: physical specs, amenity flags, geodesic distances, borough spatial join, host/market attributes | `src/feature_engineering.py` |
| 3 | Model benchmarking: Ridge vs. LightGBM/XGBoost/CatBoost with spatial (neighbourhood-grouped) 5-fold CV | `notebooks/model_development.ipynb` |
| 4 | Interpretability: TreeSHAP on the winning LightGBM model + error diagnosis | `notebooks/interpretability.ipynb` |
| 5 | Interactive dashboard: dual-layer map + scenario price predictor | `src/app.py` |

Phases 1–2 are also chainable as plain scripts via `src/main.py`; phases 3–4
are notebooks (they need to be run once to produce the model artifacts phase
5 depends on); phase 5 is a Streamlit app.

## Project structure

```
airbnb_rome/
├── data/
│   ├── raw/                  # Inside Airbnb source CSVs + neighbourhoods.geojson (not tracked)
│   └── processed/            # Outputs of phases 1, 2, and the notebooks' OOF export
├── models/                   # Saved model artifacts (produced by the Phase 4 notebook)
├── notebooks/
│   ├── model_development.ipynb   # Phase 3
│   └── interpretability.ipynb    # Phase 4
├── src/
│   ├── data_clean.py         # Phase 1
│   ├── feature_engineering.py# Phase 2
│   ├── model_training.py     # Shared training/CV/metric code, imported by the notebooks and app.py
│   ├── main.py                # Runs Phase 1 + 2 end to end
│   └── app.py                 # Phase 5 Streamlit dashboard
├── .streamlit/config.toml     # Dashboard theme
└── requirements.txt
```

## Prerequisites

- Python 3.10+ (developed and tested on 3.13)
- ~2 GB free disk (the raw `calendar.csv`/`reviews.csv` exports are large)
- Either `pip` + `venv`, or `conda`/`miniconda`

## Installation

### Option A — pip + venv (recommended)

```bash
cd airbnb_rome
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — conda

Conda is used here only to manage the Python interpreter; the packages
(LightGBM, XGBoost, CatBoost, SHAP, Streamlit, etc.) are still installed via
pip from `requirements.txt` inside that environment, since that's the exact
combination this project was built and tested against.

```bash
cd airbnb_rome
conda create -n airbnb-rome python=3.13 -y
conda activate airbnb-rome

pip install --upgrade pip
pip install -r requirements.txt
```

### Register the Jupyter kernel (needed for phases 3 & 4)

```bash
python -m ipykernel install --user --name airbnb_rome --display-name "Python 3 (airbnb_rome)"
```

Then, when opening `notebooks/model_development.ipynb` or
`notebooks/interpretability.ipynb`, select the **"Python 3 (airbnb_rome)"**
kernel.

## Data setup

Place the Inside Airbnb export files for Rome in `data/raw/`:

```
data/raw/
├── listings.csv
├── calendar.csv
├── reviews.csv
├── neighbourhoods.csv
└── neighbourhoods.geojson
```

These are not committed to the repo (large files) — source them from your
own Inside Airbnb data snapshot.

## Running the pipeline

**1–2. Clean & feature-engineer the data:**

```bash
python src/main.py
```

Produces `data/processed/listings_clean.parquet` and
`data/processed/listings_features.parquet`.

**3–4. Train, benchmark, and interpret the model:**

Run the two notebooks in order (each executes top-to-bottom, e.g. via
`jupyter lab`, `jupyter notebook`, or your editor's notebook UI, using the
`airbnb_rome` kernel registered above):

1. `notebooks/model_development.ipynb` — Ridge/LightGBM/XGBoost/CatBoost
   benchmark under spatial 5-fold CV.
2. `notebooks/interpretability.ipynb` — retrains the winning LightGBM model,
   runs SHAP analysis and error diagnosis, and **saves the artifacts the
   dashboard needs**: `models/lightgbm_price_model.joblib`,
   `models/lightgbm_feature_spec.json`, and
   `data/processed/oof_predictions.parquet`.

You can also run a notebook headlessly instead of opening it in a UI:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/model_development.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/interpretability.ipynb
```

**5. Launch the dashboard:**

```bash
streamlit run src/app.py
```

Opens at `http://localhost:8501`. Requires the artifacts produced by step
3–4 above (`models/lightgbm_price_model.joblib`,
`data/processed/oof_predictions.parquet`) — the app will show a clear error
telling you which notebook to run if any are missing.

## Troubleshooting

- **"Missing required file" error in the dashboard** — run the two
  notebooks (step 3–4) first; the app needs their output artifacts.
- **Notebook kernel not found** — re-run the `ipykernel install` command
  above with the venv/conda env active.
- **LightGBM/XGBoost/CatBoost fail to install** — these ship prebuilt
  wheels for common platforms; if `pip install` tries to compile from
  source, upgrade `pip` first (`pip install --upgrade pip`) and retry.
