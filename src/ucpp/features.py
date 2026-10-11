"""Build the preprocessing + model pipelines.

Every estimator here is a single scikit-learn ``Pipeline`` that owns its own
imputation and encoding. That is the central design decision of this project:
because preprocessing lives *inside* the estimator, calling ``fit`` on the
training split is the only way to fit it, so the medians and category lists can
never be learned from the test set. It also means the saved artefact is
self-contained - the app loads one file and calls ``predict`` on raw columns.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# Categories appearing fewer than this many times in the TRAINING split are
# pooled into a single "infrequent" bucket. Two benefits: rare model families
# (some appear once) stop producing near-useless one-hot columns, and - because
# we pair it with handle_unknown="infrequent_if_exist" - any category the model
# has never seen is routed into that same bucket at predict time instead of
# raising. That is what makes the app safe against unseen input.
MIN_CATEGORY_FREQUENCY = 15


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Assemble the column-wise preprocessing.

    Parameters
    ----------
    scale_numeric:
        Standardise numeric columns. Needed by Ridge, whose penalty is scale
        sensitive; pointless for tree models, which only compare thresholds.
    """
    numeric_steps: list = [
        # Median rather than mean: engine_cc and km_driven are right-skewed, so
        # the mean is dragged upward by a few extreme listings.
        ("impute", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), config.NUMERIC_FEATURES),
            ("categorical", categorical, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # Ignore name/year so stray columns cannot leak in.
        verbose_feature_names_out=False,
    )


def _wrap_log_target(pipeline: Pipeline) -> Pipeline:
    """Train on log1p(price) and invert the prediction back to rupees.

    Prices span 30,000 to 10,000,000 rupees and are heavily right-skewed. Fitted
    on raw rupees, squared-error models spend nearly all their effort on a
    handful of luxury cars. On the log scale the model optimises *relative*
    error, which is what matters when quoting a car price, and predictions can
    never come back negative. Metrics are still reported in rupees, because
    that is the unit anyone actually cares about.
    """
    from sklearn.compose import TransformedTargetRegressor

    final = pipeline.steps[-1]
    pipeline.steps[-1] = (
        final[0],
        TransformedTargetRegressor(
            regressor=final[1], func=np.log1p, inverse_func=np.expm1
        ),
    )
    return pipeline


def build_baseline() -> Pipeline:
    """Predict the training-set median price for every car.

    The number to beat. A model that cannot beat this has learned nothing, and
    quoting an R-squared without it hides that.
    """
    return Pipeline(
        [
            ("preprocess", build_preprocessor(scale_numeric=False)),
            ("model", DummyRegressor(strategy="median")),
        ]
    )


def build_ridge() -> Pipeline:
    """Linear model with L2 regularisation, on the log target.

    Ridge over plain LinearRegression because one-hot encoding model_family
    produces many correlated, sparsely populated columns; an unpenalised fit
    gives them wild coefficients.
    """
    return _wrap_log_target(
        Pipeline(
            [
                ("preprocess", build_preprocessor(scale_numeric=True)),
                ("model", Ridge(alpha=1.0, random_state=config.RANDOM_STATE)),
            ]
        )
    )


def build_random_forest(**overrides) -> Pipeline:
    """Random Forest: bagged trees, each split on a random feature subset."""
    params = dict(
        n_estimators=300,
        min_samples_leaf=2,
        # max_features=1.0 (every feature considered at each split) is
        # scikit-learn's default for REGRESSION forests, and it matters here.
        # "sqrt" is the classification default; on our ~110-column one-hot
        # matrix it would offer each split only ~10 candidate columns, mostly
        # sparse model_family dummies, so the trees would rarely get to split
        # on vehicle_age or max_power_bhp - the two features that carry most of
        # the signal. That alone cost ~0.12 R-squared in testing.
        max_features=1.0,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    params.update(overrides)
    return _wrap_log_target(
        Pipeline(
            [
                ("preprocess", build_preprocessor(scale_numeric=False)),
                ("model", RandomForestRegressor(**params)),
            ]
        )
    )


def build_extra_trees(**overrides) -> Pipeline:
    """Extra Trees: like Random Forest but with random split thresholds.

    The extra randomness lowers variance further and trains faster, which suits
    a CPU-only project.
    """
    params = dict(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=1.0,  # see build_random_forest for why not "sqrt"
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    params.update(overrides)
    return _wrap_log_target(
        Pipeline(
            [
                ("preprocess", build_preprocessor(scale_numeric=False)),
                ("model", ExtraTreesRegressor(**params)),
            ]
        )
    )


# Registry used by the training script, so adding a candidate model is a
# one-line change rather than an edit in three places.
MODEL_BUILDERS = {
    "baseline_median": build_baseline,
    "ridge": build_ridge,
    "random_forest": build_random_forest,
    "extra_trees": build_extra_trees,
}

MODEL_DESCRIPTIONS = {
    "baseline_median": "Always predicts the training-set median price",
    "ridge": "Linear regression with L2 penalty, log-price target",
    "random_forest": "300 bagged decision trees, log-price target",
    "extra_trees": "300 randomised-split trees, log-price target",
}


def get_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Return post-encoding feature names from a fitted pipeline."""
    return list(fitted_pipeline.named_steps["preprocess"].get_feature_names_out())
