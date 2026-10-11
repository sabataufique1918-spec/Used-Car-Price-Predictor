"""Load the raw CarDekho CSV and turn it into a clean, documented table.

The raw file stores several numbers as text with units glued on ("23.4 kmpl",
"1248 CC", "74 bhp"). This module parses those, derives the modelling features,
and records every excluded row so the exclusions can be audited instead of
being taken on trust.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Columns we expect to find in "Car details v3.csv". Checked on load so a
# different file from the same Kaggle download fails loudly instead of
# producing a silently wrong model.
EXPECTED_RAW_COLUMNS = [
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats",
]


class SchemaError(ValueError):
    """Raised when an input table does not match the expected schema."""


@dataclass
class ExclusionLog:
    """Running tally of rows dropped during cleaning, with reasons."""

    rows_in: int = 0
    rows_out: int = 0
    steps: list[dict] = field(default_factory=list)

    def record(self, reason: str, removed: int, detail: str = "") -> None:
        self.steps.append(
            {"reason": reason, "rows_removed": int(removed), "detail": detail}
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_removed"] = self.rows_in - self.rows_out
        d["pct_removed"] = (
            round(100 * (self.rows_in - self.rows_out) / self.rows_in, 2)
            if self.rows_in
            else 0.0
        )
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Unit handling
# ---------------------------------------------------------------------------
def lakh_to_inr(value_in_lakh: float) -> float:
    """Convert a price quoted in lakhs to absolute rupees (1 lakh = 100,000).

    The Kaggle download bundles four files with two different price units: the
    file we train on ("Car details v3.csv") quotes absolute rupees, while
    "car data.csv" quotes lakhs. Mixing them without converting would make a
    3.35-lakh car look like a 3-rupee car.
    """
    return float(value_in_lakh) * config.RUPEES_PER_LAKH


def inr_to_lakh(value_in_inr: float) -> float:
    """Convert absolute rupees to lakhs (inverse of :func:`lakh_to_inr`)."""
    return float(value_in_inr) / config.RUPEES_PER_LAKH


def format_inr(amount: float) -> str:
    """Format rupees using the Indian digit grouping (e.g. 12,34,567)."""
    amount = float(amount)
    sign = "-" if amount < 0 else ""
    whole = f"{abs(amount):.0f}"
    if len(whole) <= 3:
        return f"{sign}₹{whole}"
    # Last three digits, then groups of two, which is the Indian convention.
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return f"{sign}₹{','.join(groups)},{tail}"


def _extract_number(value) -> float:
    """Pull the first number out of a string like '23.4 kmpl' or '1248 CC'.

    Returns NaN for missing values and for strings with no digits at all - the
    raw file contains one max_power entry that is literally ' bhp'. Those
    become NaN and are imputed later rather than deleting the whole listing.
    """
    if value is None:
        return np.nan
    if isinstance(value, float) and np.isnan(value):
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Read the raw CSV and verify it has the schema this project expects."""
    path = Path(path) if path is not None else config.RAW_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            f"Run:  python scripts/download_data.py\n"
            f"Source: {config.KAGGLE_DATASET_URL}"
        )
    df = pd.read_csv(path)
    validate_raw_schema(df)
    return df


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Raise :class:`SchemaError` if required raw columns are missing.

    This is the guard that catches someone pointing the pipeline at
    "car data.csv", whose columns are title-cased AND whose prices are in
    lakhs - a mistake that would otherwise train a model on nonsense.
    """
    missing = [c for c in EXPECTED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Raw data is missing expected column(s): {missing}. "
            f"Expected the schema of '{config.RAW_CSV_NAME}' with columns "
            f"{EXPECTED_RAW_COLUMNS}. Got: {list(df.columns)}"
        )


def validate_feature_frame(df: pd.DataFrame) -> None:
    """Raise :class:`SchemaError` unless every modelling feature is present.

    Called before predicting so a malformed request fails with a clear message
    instead of reaching scikit-learn and raising something cryptic.
    """
    missing = [c for c in config.FEATURES if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Input is missing required feature(s): {missing}. "
            f"Required features: {config.FEATURES}"
        )


# ---------------------------------------------------------------------------
# Feature derivation
# ---------------------------------------------------------------------------
def derive_features(
    df: pd.DataFrame, observation_year: int | None = None
) -> pd.DataFrame:
    """Parse the unit-suffixed text columns and derive brand/model/age.

    Pure function: returns a new frame and never mutates the input.
    """
    observation_year = observation_year or config.OBSERVATION_YEAR
    out = df.copy()

    # Numbers stored as text with units.
    out["mileage_kmpl"] = out["mileage"].map(_extract_number)
    out["engine_cc"] = out["engine"].map(_extract_number)
    out["max_power_bhp"] = out["max_power"].map(_extract_number)
    out["seats"] = pd.to_numeric(out["seats"], errors="coerce")

    # 'torque' is deliberately NOT used as a feature. Its values mix units and
    # rev ranges ("190Nm@ 2000rpm", "22.4 kgm at 1750-2750rpm"), so parsing it
    # into one comparable number is guesswork, and it largely duplicates
    # information already carried by engine_cc and max_power_bhp.

    # Age relative to the dataset's observation period, not today. See
    # config.OBSERVATION_YEAR for why this matters.
    out["vehicle_age"] = observation_year - pd.to_numeric(out["year"], errors="coerce")

    # 'name' is "Maruti Swift Dzire VDI": first token is the brand, first two
    # tokens the model family. Full names have 2,058 distinct values, which is
    # too granular to learn from ~6,900 rows; model_family has 210.
    name = out["name"].astype("string").str.strip()
    parts = name.str.split()
    out["brand"] = parts.str[0]
    out["model_family"] = parts.str[:2].str.join(" ")

    return out


def clean(
    df: pd.DataFrame,
    observation_year: int | None = None,
    log: ExclusionLog | None = None,
) -> tuple[pd.DataFrame, ExclusionLog]:
    """Clean the raw frame, returning the clean table and an exclusion log.

    Exclusions are limited to records that are duplicated or physically
    impossible. Expensive cars, very old cars and very high-mileage cars are
    all kept, because dropping them would make the reported error look better
    without the model actually being better.
    """
    observation_year = observation_year or config.OBSERVATION_YEAR
    log = log or ExclusionLog()
    log.rows_in = len(df)

    work = df.copy()

    # 1. Exact duplicate rows. The scrape republished ~15% of listings
    #    verbatim. These MUST go before splitting: an identical row in both
    #    train and test is a memorisation test, not a generalisation test.
    before = len(work)
    work = work.drop_duplicates()
    log.record(
        "exact_duplicate_rows",
        before - len(work),
        "Byte-identical listings in the raw scrape. Removed before splitting so "
        "the same listing cannot appear in both train and test.",
    )

    # 2. Missing or unusable target.
    before = len(work)
    work = work[pd.to_numeric(work[config.TARGET], errors="coerce").notna()]
    log.record(
        "missing_price", before - len(work), "selling_price absent or non-numeric."
    )

    work[config.TARGET] = pd.to_numeric(work[config.TARGET])

    before = len(work)
    work = work[
        work[config.TARGET].between(
            config.MIN_PLAUSIBLE_PRICE_INR, config.MAX_PLAUSIBLE_PRICE_INR
        )
    ]
    log.record(
        "implausible_price",
        before - len(work),
        f"selling_price outside "
        f"{config.MIN_PLAUSIBLE_PRICE_INR:,}-{config.MAX_PLAUSIBLE_PRICE_INR:,} INR.",
    )

    # 3. Model year must be usable and must not post-date the observation
    #    period, which would produce a negative age.
    before = len(work)
    year = pd.to_numeric(work["year"], errors="coerce")
    work = work[year.between(config.MIN_PLAUSIBLE_YEAR, config.MAX_PLAUSIBLE_YEAR)]
    log.record(
        "implausible_year",
        before - len(work),
        f"Model year outside {config.MIN_PLAUSIBLE_YEAR}-{config.MAX_PLAUSIBLE_YEAR} "
        f"(upper bound is the dataset's observation year).",
    )

    # 4. Odometer must be present, non-negative and physically possible.
    before = len(work)
    km = pd.to_numeric(work["km_driven"], errors="coerce")
    work = work[km.notna() & (km >= 0)]
    log.record(
        "missing_or_negative_km", before - len(work), "km_driven absent or negative."
    )

    work = derive_features(work, observation_year=observation_year)

    before = len(work)
    # Age+1 so a car from the observation year itself divides by 1, not 0.
    km_per_year = work["km_driven"] / (work["vehicle_age"] + 1)
    work = work[km_per_year <= config.MAX_PLAUSIBLE_KM_PER_YEAR]
    log.record(
        "impossible_mileage_rate",
        before - len(work),
        f"Implied usage above {config.MAX_PLAUSIBLE_KM_PER_YEAR:,} km/year "
        f"(~410 km every day of the vehicle's life). Genuinely high-mileage "
        f"cars below this rate are kept.",
    )

    keep = config.FEATURES + [config.TARGET, "name", "year"]
    work = work[keep].reset_index(drop=True)

    log.rows_out = len(work)
    return work, log


def vehicle_identity(df: pd.DataFrame) -> pd.Series:
    """Build a grouping key identifying (probably) the same physical vehicle.

    After exact duplicates are removed, hundreds of near-identical groups
    remain: the same name, year and odometer reading listed at slightly
    different prices. These are almost certainly one car relisted. Grouping on
    this key and splitting by group stops a car's twin sitting in the test set.
    """
    return (
        df["name"].astype("string").str.strip().str.lower()
        + "|"
        + df["year"].astype(str)
        + "|"
        + df["km_driven"].astype(str)
    )


def load_clean(path: Path | str | None = None) -> pd.DataFrame:
    """Load the prepared table written by ``scripts/prepare_data.py``."""
    path = Path(path) if path is not None else config.CLEAN_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found at {path}.\n"
            f"Run:  python scripts/prepare_data.py"
        )
    return pd.read_csv(path)
