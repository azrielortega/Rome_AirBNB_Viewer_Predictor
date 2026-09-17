"""
Pipeline entry point: runs Phase 1 (ingestion & cleaning) followed by
Phase 2 (feature engineering) on the Airbnb Rome dataset.

Usage:
    python src/main.py
"""

import data_clean
import feature_engineering

def main() -> None:
    print("=" * 80)
    print("PHASE 1: Ingestion, Scrubbing & Data Hygiene")
    print("=" * 80)
    cleaned_df = data_clean.run()

    print()
    print("=" * 80)
    print("PHASE 2: Feature Engineering (Physical & Spatial)")
    print("=" * 80)
    feature_engineering.run(cleaned_df)

if __name__ == "__main__":
    main()
