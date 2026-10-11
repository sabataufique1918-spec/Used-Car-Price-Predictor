"""Clean the raw CSV and write data/processed/cars_clean.csv.

Separate from training on purpose: cleaning decisions are data decisions and
should be reviewable (and diffable) on their own, without rerunning a fit.

Usage:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ucpp import config, data_prep  # noqa: E402
from ucpp.console import enable_utf8_stdout  # noqa: E402

enable_utf8_stdout()


def main() -> None:
    print("=" * 72)
    print("PREPARING DATA")
    print("=" * 72)

    raw = data_prep.load_raw()
    print(f"\nLoaded {config.RAW_CSV_PATH.name}: {raw.shape[0]:,} rows x "
          f"{raw.shape[1]} columns")

    # --- Raw data audit, printed so the numbers in the README are traceable.
    print("\n--- Missing values in raw data ---")
    missing = raw.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  none")
    for col, count in missing.items():
        print(f"  {col:<14} {count:>5,}  ({100 * count / len(raw):.1f}%)")

    print("\n--- Duplicates in raw data ---")
    print(f"  exact duplicate rows:              {raw.duplicated().sum():,}")
    print(
        "  duplicate on all columns but price: "
        f"{raw.drop(columns=[config.TARGET]).duplicated().sum():,}"
    )

    print("\n--- Target (selling_price) in raw data ---")
    print(f"  units: absolute INR (verified: min {raw[config.TARGET].min():,}, "
          f"max {raw[config.TARGET].max():,})")
    print(f"  NOTE: sibling file 'car data.csv' quotes LAKHS and is not used.")
    print(f"  observation period ends: year {raw['year'].max()} "
          f"-> config.OBSERVATION_YEAR = {config.OBSERVATION_YEAR}")
    if raw["year"].max() != config.OBSERVATION_YEAR:
        print(
            f"  WARNING: config.OBSERVATION_YEAR ({config.OBSERVATION_YEAR}) does "
            f"not match the data's max year ({raw['year'].max()}). Update config."
        )

    # --- Clean.
    clean_df, log = data_prep.clean(raw)

    print("\n--- Exclusions ---")
    for step in log.steps:
        if step["rows_removed"]:
            print(f"  {step['rows_removed']:>5,}  {step['reason']}")
            print(f"         {step['detail']}")
    print(f"\n  rows in:  {log.rows_in:,}")
    print(f"  rows out: {log.rows_out:,}")
    print(f"  removed:  {log.rows_in - log.rows_out:,} "
          f"({log.to_dict()['pct_removed']}%)")

    # --- Residual missingness after parsing: imputed inside the pipeline, not
    #     dropped here, so no listing is lost over one absent spec value.
    print("\n--- Missing feature values after parsing (imputed in-pipeline) ---")
    feat_missing = clean_df[config.FEATURES].isna().sum()
    feat_missing = feat_missing[feat_missing > 0]
    if feat_missing.empty:
        print("  none")
    for col, count in feat_missing.items():
        print(f"  {col:<16} {count:>5,}  ({100 * count / len(clean_df):.1f}%)")

    # --- Leakage exposure the group split has to handle.
    identity = data_prep.vehicle_identity(clean_df)
    counts = identity.value_counts()
    print("\n--- Near-duplicate vehicles (same name + year + odometer) ---")
    print(f"  distinct vehicles:        {identity.nunique():,}")
    print(f"  vehicles listed >1 time:  {(counts > 1).sum():,}")
    print(f"  rows in such groups:      {counts[counts > 1].sum():,}")
    print("  -> train/test split is grouped on this key (see train_model.py)")

    print("\n--- Clean dataset summary ---")
    print(f"  rows: {len(clean_df):,}")
    print(f"  price (INR): min {clean_df[config.TARGET].min():,.0f} | "
          f"median {clean_df[config.TARGET].median():,.0f} | "
          f"max {clean_df[config.TARGET].max():,.0f}")
    print(f"  vehicle_age: {clean_df['vehicle_age'].min():.0f} to "
          f"{clean_df['vehicle_age'].max():.0f} years "
          f"(relative to {config.OBSERVATION_YEAR})")
    print(f"  brands: {clean_df['brand'].nunique()} | "
          f"model families: {clean_df['model_family'].nunique()}")

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(config.CLEAN_CSV_PATH, index=False)
    log.save(config.EXCLUSION_LOG_PATH)

    print(f"\nWrote {config.CLEAN_CSV_PATH}")
    print(f"Wrote {config.EXCLUSION_LOG_PATH}")
    print("\nNext:  python scripts/train_model.py")


if __name__ == "__main__":
    main()
