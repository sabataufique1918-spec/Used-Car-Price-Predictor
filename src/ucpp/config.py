"""Central configuration: paths, dataset facts, and the feature schema.

Everything that another module might otherwise hard-code lives here, so there
is exactly one place to look when a path or a column name changes.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths. PROJECT_ROOT is resolved from this file's location (src/ucpp/config.py
# -> up three parents), so every script works no matter which directory it is
# launched from. That matters for Streamlit Cloud, which sets its own CWD.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# The one dataset file this project trains on. See README "Why this file".
RAW_CSV_NAME = "Car details v3.csv"
RAW_CSV_PATH = RAW_DATA_DIR / RAW_CSV_NAME
CLEAN_CSV_PATH = PROCESSED_DATA_DIR / "cars_clean.csv"
EXCLUSION_LOG_PATH = PROCESSED_DATA_DIR / "exclusions.json"

MODEL_PATH = MODELS_DIR / "price_pipeline.joblib"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
METRICS_PATH = REPORTS_DIR / "evaluation.json"

# --------------------------------------------------------------------------
# Dataset facts (verified against the raw file, not assumed).
# --------------------------------------------------------------------------
KAGGLE_DATASET_REF = "nehalbirla/vehicle-dataset-from-cardekho"
KAGGLE_DATASET_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET_REF}"
DATASET_LICENCE = "Open Database License (ODbL) v1.0"

# The newest model year present in "Car details v3.csv" is 2020, and the
# dataset was last published in January 2023. Vehicle age is therefore measured
# against 2020 - NOT against the current calendar year. Using today's year
# would silently add years of phantom age to every car and push predictions
# outside the range the model was ever fitted on.
OBSERVATION_YEAR = 2020

# Model years outside this range are treated as data errors. The lower bound is
# the oldest plausible listing; the upper bound is the observation year.
MIN_PLAUSIBLE_YEAR = 1980
MAX_PLAUSIBLE_YEAR = OBSERVATION_YEAR

# 150,000 km/year is ~410 km every single day of the vehicle's life. Records
# above this are physically impossible rather than merely unusual, so they are
# excluded. Genuinely high-mileage cars are deliberately KEPT: removing hard
# examples to flatter the metrics would make the evaluation dishonest.
MAX_PLAUSIBLE_KM_PER_YEAR = 150_000

# Price sanity bounds in INR. These are wide on purpose - the ten-million-rupee
# Volvo XC90 in the data is a real listing and stays in the training set.
MIN_PLAUSIBLE_PRICE_INR = 10_000
MAX_PLAUSIBLE_PRICE_INR = 50_000_000

# One lakh = 100,000 rupees. Needed because sibling files in the same Kaggle
# download quote prices in lakhs while the file we use quotes absolute rupees.
RUPEES_PER_LAKH = 100_000

# --------------------------------------------------------------------------
# Feature schema. The app, the training script and the tests all read these
# lists, which is what keeps the Streamlit form and the fitted pipeline from
# drifting apart.
# --------------------------------------------------------------------------
TARGET = "selling_price"

NUMERIC_FEATURES = [
    "vehicle_age",
    "km_driven",
    "mileage_kmpl",
    "engine_cc",
    "max_power_bhp",
    "seats",
]

CATEGORICAL_FEATURES = [
    "brand",
    "model_family",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Plain-language labels and help text, used to build the Streamlit form.
FEATURE_LABELS = {
    "brand": "Brand",
    "model_family": "Model",
    "vehicle_age": "Vehicle age (years)",
    "km_driven": "Kilometres driven",
    "fuel": "Fuel type",
    "seller_type": "Seller type",
    "transmission": "Transmission",
    "owner": "Ownership history",
    "mileage_kmpl": "Mileage (kmpl)",
    "engine_cc": "Engine displacement (CC)",
    "max_power_bhp": "Max power (bhp)",
    "seats": "Seats",
}

# Price bands used for the error breakdown. Bounds are in INR and chosen from
# the observed price distribution rather than round numbers for their own sake.
PRICE_BANDS = [
    ("Under 2 lakh", 0, 200_000),
    ("2-4 lakh", 200_000, 400_000),
    ("4-7 lakh", 400_000, 700_000),
    ("7-12 lakh", 700_000, 1_200_000),
    ("Above 12 lakh", 1_200_000, float("inf")),
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
