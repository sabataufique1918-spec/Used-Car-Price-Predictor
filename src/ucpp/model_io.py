"""Save and load the trained pipeline plus its feature schema.

Security note: joblib files are pickles, and unpickling runs arbitrary code.
This project therefore loads models from ONE place only - the project's own
``models/`` directory, produced by ``scripts/train_model.py``. The app has no
model-upload control and never will. If you cannot point at the script that
generated an artefact, do not load it.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from . import config


class SchemaMismatchError(RuntimeError):
    """Raised when a loaded model's schema disagrees with the running code."""


def build_schema(
    clean_df: pd.DataFrame, model_name: str, metrics: dict | None = None
) -> dict:
    """Describe the inputs the saved model expects.

    Saved alongside the pipeline so the app can build its form, validate
    requests and populate dependent dropdowns without ever reading the training
    CSV. This is what lets the deployed app run from the model file alone.
    """
    categories = {
        col: sorted(clean_df[col].dropna().astype(str).unique().tolist())
        for col in config.CATEGORICAL_FEATURES
    }

    # Most common value per categorical column. The app uses these as form
    # defaults: sorting alphabetically would default Fuel to "CNG" (57 of 6,924
    # listings) and Transmission to "Automatic", so the form would open on a
    # rare combination and quietly mislead anyone who did not change it.
    category_modes = {
        col: str(clean_df[col].dropna().astype(str).mode().iloc[0])
        for col in config.CATEGORICAL_FEATURES
        if clean_df[col].notna().any()
    }

    # Counts per category, so the app can show how much support a choice has.
    category_counts = {
        col: clean_df[col].dropna().astype(str).value_counts().to_dict()
        for col in config.CATEGORICAL_FEATURES
    }

    numeric_ranges = {}
    for col in config.NUMERIC_FEATURES:
        series = pd.to_numeric(clean_df[col], errors="coerce").dropna()
        numeric_ranges[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "p01": float(series.quantile(0.01)),
            "p99": float(series.quantile(0.99)),
        }

    # Which model families belong to which brand - drives dependent dropdowns.
    brand_to_models = (
        clean_df.groupby("brand")["model_family"]
        .apply(lambda s: sorted(s.dropna().astype(str).unique().tolist()))
        .to_dict()
    )

    return {
        "schema_version": 1,
        "project_version": config.__dict__.get("__version__", "1.0.0"),
        "model_name": model_name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": config.TARGET,
        "target_units": "INR (absolute rupees, not lakhs)",
        "observation_year": config.OBSERVATION_YEAR,
        "features": config.FEATURES,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "categories": categories,
        "category_modes": category_modes,
        "category_counts": category_counts,
        "numeric_ranges": numeric_ranges,
        "brand_to_models": brand_to_models,
        "training_rows": int(len(clean_df)),
        "price_range_inr": {
            "min": float(clean_df[config.TARGET].min()),
            "max": float(clean_df[config.TARGET].max()),
            "median": float(clean_df[config.TARGET].median()),
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "metrics": metrics or {},
        "source": {
            "dataset": config.KAGGLE_DATASET_REF,
            "file": config.RAW_CSV_NAME,
            "licence": config.DATASET_LICENCE,
        },
    }


def save_model(
    pipeline, schema: dict, model_path: Path | None = None, schema_path: Path | None = None
) -> tuple[Path, Path]:
    """Persist the fitted pipeline and its schema."""
    model_path = Path(model_path or config.MODEL_PATH)
    schema_path = Path(schema_path or config.SCHEMA_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path, compress=3)
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return model_path, schema_path


def load_model(
    model_path: Path | None = None, schema_path: Path | None = None
) -> tuple[object, dict]:
    """Load the project's trained pipeline and schema.

    Raises a clear, actionable error if the artefacts are absent, and refuses a
    model whose schema does not match the feature list the current code uses -
    silently predicting from mismatched columns is worse than failing.
    """
    model_path = Path(model_path or config.MODEL_PATH)
    schema_path = Path(schema_path or config.SCHEMA_PATH)

    if not model_path.exists() or not schema_path.exists():
        raise FileNotFoundError(
            "Trained model artefacts not found.\n"
            f"  expected model:  {model_path}\n"
            f"  expected schema: {schema_path}\n"
            "Train the model first:\n"
            "  python scripts/download_data.py\n"
            "  python scripts/prepare_data.py\n"
            "  python scripts/train_model.py"
        )

    pipeline = joblib.load(model_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    expected = list(config.FEATURES)
    found = list(schema.get("features", []))
    if found != expected:
        raise SchemaMismatchError(
            "Saved model schema does not match the current feature definition.\n"
            f"  model was trained on: {found}\n"
            f"  code expects:         {expected}\n"
            "Retrain with: python scripts/train_model.py"
        )

    return pipeline, schema
