"""Metrics and error breakdowns, all reported in rupees.

Kept separate from training so the app can re-read the saved numbers without
importing anything that fits a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from . import config


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Core metrics for one set of predictions.

    MAE and RMSE are both reported because they answer different questions.
    MAE is the typical miss in rupees. RMSE squares errors before averaging, so
    it is much more sensitive to the occasional badly wrong luxury-car quote. A
    large RMSE/MAE gap is itself a finding: it means most predictions are
    decent but a few are far off.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_pred - y_true
    # Median absolute PERCENTAGE error: scale-free, and the median keeps it
    # from being dominated by the cheapest cars, where a small rupee error is a
    # large percentage one.
    mape_median = float(np.median(np.abs(errors) / np.maximum(y_true, 1)) * 100)

    return {
        "mae_inr": float(mean_absolute_error(y_true, y_pred)),
        "rmse_inr": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "median_abs_pct_error": mape_median,
        "median_error_inr": float(np.median(errors)),
        "n": int(len(y_true)),
    }


def error_by_price_band(y_true, y_pred) -> pd.DataFrame:
    """Break error down by actual price band.

    A single headline MAE hides the model's real behaviour: error in rupees
    grows with price almost mechanically. Splitting by band shows where the
    model is actually usable and where it is not, which belongs in front of
    anyone reading a prediction.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rows = []
    for label, low, high in config.PRICE_BANDS:
        mask = (y_true >= low) & (y_true < high)
        if not mask.any():
            continue
        bt, bp = y_true[mask], y_pred[mask]
        abs_err = np.abs(bp - bt)
        rows.append(
            {
                "price_band": label,
                "n": int(mask.sum()),
                "share_of_test_pct": round(100 * mask.sum() / len(y_true), 1),
                "mae_inr": float(abs_err.mean()),
                "rmse_inr": float(np.sqrt(((bp - bt) ** 2).mean())),
                "median_abs_pct_error": float(
                    np.median(abs_err / np.maximum(bt, 1)) * 100
                ),
                "mean_bias_inr": float((bp - bt).mean()),
            }
        )
    return pd.DataFrame(rows)


def permutation_importance_frame(
    pipeline, X: pd.DataFrame, y, n_repeats: int = 5, random_state: int | None = None
) -> pd.DataFrame:
    """Rank input features by how much shuffling each one hurts performance.

    Measured on the raw input columns rather than the one-hot encoded ones, so
    "model_family" is scored as one feature instead of 70-odd dummies. Reported
    instead of tree ``feature_importances_`` because impurity importance is
    biased towards high-cardinality features - exactly what model_family is.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state if random_state is not None else config.RANDOM_STATE,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    # With scoring="neg_mean_absolute_error", importances_mean is already
    # (baseline_score - permuted_score). Shuffling a useful column raises MAE,
    # which lowers neg-MAE, so the value is positive and reads directly as
    # "rupees of MAE added by destroying this column". No sign flip needed.
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "mae_increase_inr": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("mae_increase_inr", ascending=False)
        .reset_index(drop=True)
    )


def improvement_over_baseline(model_mae: float, baseline_mae: float) -> float:
    """Percentage reduction in MAE relative to the median baseline."""
    if baseline_mae <= 0:
        return 0.0
    return float(100 * (baseline_mae - model_mae) / baseline_mae)
