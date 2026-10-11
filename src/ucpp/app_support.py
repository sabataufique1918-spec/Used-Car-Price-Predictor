"""Cached loaders and shared helpers for the Streamlit app.

Kept out of the page modules so every page shares one cached copy of the model
and the data, and so the pages stay mostly layout code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from . import config, data_prep, model_io

# Plotly colours used across pages, defined once so the charts look like one
# app rather than four.
ACCENT = "#2563eb"
ACCENT_SOFT = "#93c5fd"
WARN = "#d97706"
NEUTRAL = "#64748b"
PLOT_TEMPLATE = "plotly_white"


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    """Load the trained pipeline and schema once per server process.

    ``cache_resource`` (not ``cache_data``) because a fitted pipeline is a live
    object that must not be copied per session. Training never happens here -
    the app only ever reads artefacts that ``scripts/train_model.py`` produced.
    """
    return model_io.load_model()


@st.cache_data(show_spinner="Loading data...")
def get_clean_data() -> pd.DataFrame:
    """Load the prepared dataset, rebuilding it in memory if necessary.

    ``data/processed/`` is gitignored (it is derived, not source), but the raw
    CSV is committed. On a fresh clone - and on Streamlit Community Cloud,
    which installs requirements and runs the app but never runs the training
    scripts - the processed file is therefore absent and the Data Explorer
    would crash.

    Cleaning is deterministic and takes under a second, so the fallback
    reproduces it from the raw file instead. Same function the prepare script
    calls, so the result is identical.
    """
    try:
        return data_prep.load_clean()
    except FileNotFoundError:
        clean_df, _ = data_prep.clean(data_prep.load_raw())
        return clean_df


@st.cache_data(show_spinner=False)
def get_evaluation_report() -> dict | None:
    """Load the saved evaluation report, or None if training has not been run.

    The app never recomputes metrics. Every number on the Model Evaluation page
    comes from this file, written during training against the held-out test set.
    """
    path = Path(config.METRICS_PATH)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_exclusion_log() -> dict | None:
    """Load the cleaning exclusion log, recomputing it if the file is absent.

    Same reasoning as :func:`get_clean_data`: the log is derived and therefore
    gitignored, but the Data Explorer's provenance panel should still work on
    a fresh clone or a cloud deployment.
    """
    path = Path(config.EXCLUSION_LOG_PATH)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        _, log = data_prep.clean(data_prep.load_raw())
        return log.to_dict()
    except FileNotFoundError:
        return None


def artefacts_available() -> bool:
    """True when both model artefacts exist on disk."""
    return Path(config.MODEL_PATH).exists() and Path(config.SCHEMA_PATH).exists()


def render_missing_artefacts() -> None:
    """Explain how to train, when the app is started without a model."""
    st.error("No trained model found.")
    st.markdown(
        f"""
This app loads a model produced by the project's own training script. It does
not train at startup, and it cannot accept an uploaded model file.

Run these three commands from the project root, then reload:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train_model.py
```

Expected artefacts:

- `{Path(config.MODEL_PATH).relative_to(config.PROJECT_ROOT)}`
- `{Path(config.SCHEMA_PATH).relative_to(config.PROJECT_ROOT)}`
"""
    )


def format_inr(amount: float) -> str:
    """Indian-grouped rupee formatting (re-exported for the page modules)."""
    return data_prep.format_inr(amount)


def format_lakh(amount: float) -> str:
    """Render rupees as lakhs, the unit Indian car buyers actually use."""
    return f"{data_prep.inr_to_lakh(amount):.2f} lakh"


HISTORICAL_NOTICE = (
    f"This model is trained on used-car listings scraped from CarDekho, with "
    f"the newest model year being {config.OBSERVATION_YEAR}. It reproduces "
    f"**asking prices from that period** - it is not a current market "
    f"valuation and has not been checked against today's prices."
)
