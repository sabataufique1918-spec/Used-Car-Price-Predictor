"""Turn user input into a validated prediction, with honest warnings.

The app calls :func:`predict_price` and renders whatever warnings come back.
Keeping this logic out of the Streamlit layer means the same checks are
exercised by the tests, and the rules live in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config
from .data_prep import validate_feature_frame


@dataclass
class PredictionResult:
    """A price estimate plus everything the caller should disclose with it."""

    price_inr: float
    warnings: list[str] = field(default_factory=list)
    unseen_categories: dict[str, str] = field(default_factory=dict)
    out_of_range: dict[str, dict] = field(default_factory=dict)

    @property
    def is_extrapolation(self) -> bool:
        """True when the inputs sit outside what the model learned from."""
        return bool(self.unseen_categories or self.out_of_range)


def build_input_frame(values: dict) -> pd.DataFrame:
    """Build a one-row frame with exactly the model's feature columns.

    Missing keys become NaN rather than raising: the pipeline's imputers are
    fitted and can fill them. Extra keys are dropped so a stray column cannot
    change the column order the pipeline expects.
    """
    row = {feat: values.get(feat, None) for feat in config.FEATURES}
    frame = pd.DataFrame([row], columns=config.FEATURES)

    for col in config.NUMERIC_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in config.CATEGORICAL_FEATURES:
        frame[col] = frame[col].astype("object")

    return frame


def check_categories(frame: pd.DataFrame, schema: dict) -> dict[str, str]:
    """Find categorical values the model never saw in training.

    The encoder is configured with ``handle_unknown="infrequent_if_exist"``, so
    these do not crash - they are routed to the "infrequent" bucket. The
    prediction is still weaker than usual, so we surface it rather than letting
    the user assume the estimate is as reliable as any other.
    """
    known = schema.get("categories", {})
    unseen = {}
    for col in config.CATEGORICAL_FEATURES:
        value = frame.at[0, col]
        if value is None or pd.isna(value):
            continue
        if col in known and str(value) not in known[col]:
            unseen[col] = str(value)
    return unseen


def check_numeric_ranges(frame: pd.DataFrame, schema: dict) -> dict[str, dict]:
    """Flag numeric inputs outside the training data's 1st-99th percentile.

    Tree models cannot extrapolate: feed a Random Forest 900,000 km and it
    returns whatever it learned at the highest odometer reading it ever saw. The
    answer looks normal, so the caller has to be told.
    """
    ranges = schema.get("numeric_ranges", {})
    flagged = {}
    for col in config.NUMERIC_FEATURES:
        value = frame.at[0, col]
        if value is None or pd.isna(value) or col not in ranges:
            continue
        lo, hi = ranges[col]["p01"], ranges[col]["p99"]
        value = float(value)
        if value < lo or value > hi:
            flagged[col] = {
                "value": value,
                "typical_min": lo,
                "typical_max": hi,
                "direction": "below" if value < lo else "above",
            }
    return flagged


def predict_price(pipeline, schema: dict, values: dict) -> PredictionResult:
    """Predict one car's price in rupees and report input-quality caveats.

    No confidence interval is returned. A single Random Forest point prediction
    carries no calibrated uncertainty, and inventing a "+/- 10%" band or a
    per-prediction accuracy figure would be making up a number. What the app
    shows instead is the model's measured error on the held-out test set,
    broken down by price band.
    """
    frame = build_input_frame(values)
    validate_feature_frame(frame)

    unseen = check_categories(frame, schema)
    out_of_range = check_numeric_ranges(frame, schema)

    price = float(pipeline.predict(frame)[0])

    warnings: list[str] = []
    for col, value in unseen.items():
        label = config.FEATURE_LABELS.get(col, col)
        warnings.append(
            f"'{value}' is not a {label.lower()} seen during training. It was "
            f"grouped with rare categories, so this estimate is less reliable."
        )
    for col, info in out_of_range.items():
        label = config.FEATURE_LABELS.get(col, col)
        warnings.append(
            f"{label} of {info['value']:,.0f} is {info['direction']} the typical "
            f"training range ({info['typical_min']:,.0f} to "
            f"{info['typical_max']:,.0f}). Tree models do not extrapolate, so "
            f"the estimate is clamped to the edge of what the model has seen."
        )

    # A price outside the observed target range is a strong signal that the
    # inputs are unusual, independently of the per-feature checks above.
    price_range = schema.get("price_range_inr", {})
    if price_range and not (
        price_range["min"] * 0.5 <= price <= price_range["max"] * 1.5
    ):
        warnings.append(
            "The estimate falls outside the price range present in the training "
            "data. Treat it as unreliable."
        )

    if price < 0:
        # Should be impossible: the log-target transform bounds output above
        # zero. Guarded anyway so the app never displays a negative price.
        price = 0.0
        warnings.append("Model returned a negative value; clamped to zero.")

    return PredictionResult(
        price_inr=price,
        warnings=warnings,
        unseen_categories=unseen,
        out_of_range=out_of_range,
    )
