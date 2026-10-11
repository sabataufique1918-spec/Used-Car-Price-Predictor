"""Tests for the pipeline: leakage containment, unknown categories, persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

from ucpp import config, data_prep, evaluate, features, model_io


class TestPreprocessing:
    def test_preprocessor_imputes_missing_numeric_values(self, trainable_frame):
        pre = features.build_preprocessor(scale_numeric=False)
        frame = trainable_frame[config.FEATURES].copy()
        frame.loc[0, "max_power_bhp"] = np.nan
        transformed = pre.fit_transform(frame)
        assert not np.isnan(transformed).any()

    def test_preprocessing_is_fitted_only_on_what_it_is_given(self, trainable_frame):
        """Imputation values must come from the fitting data alone.

        This is the guarantee that makes a Pipeline safe inside
        cross-validation: fitting on a subset must not peek at the rest.
        """
        pre = features.build_preprocessor(scale_numeric=False)
        subset = trainable_frame[config.FEATURES].head(40).copy()
        pre.fit(subset)
        learned = pre.named_transformers_["numeric"].named_steps["impute"].statistics_
        expected = subset[config.NUMERIC_FEATURES].median().to_numpy(dtype=float)
        np.testing.assert_allclose(learned, expected)

    def test_numeric_scaling_only_applied_when_requested(self, trainable_frame):
        unscaled = features.build_preprocessor(scale_numeric=False)
        scaled = features.build_preprocessor(scale_numeric=True)
        frame = trainable_frame[config.FEATURES]

        def numeric_steps(preprocessor):
            # ColumnTransformer.transformers holds (name, transformer, columns).
            return next(
                transformer.named_steps
                for name, transformer, _ in preprocessor.transformers
                if name == "numeric"
            )

        assert "scale" not in numeric_steps(unscaled)
        assert "scale" in numeric_steps(scaled)
        # Scaled numeric block should be roughly zero-mean.
        out = scaled.fit_transform(frame)
        assert abs(out[:, : len(config.NUMERIC_FEATURES)].mean()) < 1e-9


class TestUnknownCategories:
    """The app must never crash on a category the model has not seen."""

    def test_unseen_category_does_not_raise(self, fitted_pipeline_and_schema):
        pipeline, _ = fitted_pipeline_and_schema
        row = _example_row()
        row["brand"] = "Lamborghini"          # never in training
        row["model_family"] = "Lamborghini Urus"
        prediction = pipeline.predict(pd.DataFrame([row]))
        assert len(prediction) == 1
        assert np.isfinite(prediction[0])

    def test_unseen_values_in_every_categorical_column(self, fitted_pipeline_and_schema):
        pipeline, _ = fitted_pipeline_and_schema
        for column in config.CATEGORICAL_FEATURES:
            row = _example_row()
            row[column] = "A Value Never Seen Before"
            prediction = pipeline.predict(pd.DataFrame([row]))
            assert np.isfinite(prediction[0]), f"failed on {column}"

    def test_missing_categorical_value_is_imputed(self, fitted_pipeline_and_schema):
        pipeline, _ = fitted_pipeline_and_schema
        row = _example_row()
        row["fuel"] = None
        assert np.isfinite(pipeline.predict(pd.DataFrame([row]))[0])

    def test_predictions_are_never_negative(self, fitted_pipeline_and_schema):
        """The log-target transform must keep output above zero.

        A negative price would be visibly nonsense in the app, so this is
        checked against deliberately extreme inputs.
        """
        pipeline, _ = fitted_pipeline_and_schema
        for age, km in ((40, 900_000), (0, 0), (25, 500_000)):
            row = _example_row()
            row["vehicle_age"] = age
            row["km_driven"] = km
            assert pipeline.predict(pd.DataFrame([row]))[0] > 0


class TestModelPersistence:
    def test_save_then_reload_gives_identical_predictions(
        self, fitted_pipeline_and_schema, tmp_path
    ):
        """A reloaded model must behave exactly like the one that was saved."""
        pipeline, schema = fitted_pipeline_and_schema
        model_path = tmp_path / "pipeline.joblib"
        schema_path = tmp_path / "schema.json"

        sample = pd.DataFrame([_example_row() for _ in range(5)])
        before = pipeline.predict(sample)

        model_io.save_model(pipeline, schema, model_path, schema_path)
        assert model_path.exists() and schema_path.exists()

        reloaded, reloaded_schema = model_io.load_model(model_path, schema_path)
        after = reloaded.predict(sample)

        np.testing.assert_array_almost_equal(before, after, decimal=6)
        assert reloaded_schema["features"] == config.FEATURES

    def test_load_raises_a_helpful_error_when_absent(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train_model.py"):
            model_io.load_model(tmp_path / "nope.joblib", tmp_path / "nope.json")

    def test_load_rejects_a_schema_that_does_not_match_the_code(
        self, fitted_pipeline_and_schema, tmp_path
    ):
        """Predicting from mismatched columns is worse than failing loudly."""
        pipeline, schema = fitted_pipeline_and_schema
        stale = dict(schema)
        stale["features"] = ["brand", "vehicle_age"]  # an older feature set

        model_path = tmp_path / "pipeline.joblib"
        schema_path = tmp_path / "schema.json"
        model_io.save_model(pipeline, stale, model_path, schema_path)

        with pytest.raises(model_io.SchemaMismatchError, match="Retrain"):
            model_io.load_model(model_path, schema_path)

    def test_schema_records_units_and_observation_year(self, fitted_pipeline_and_schema):
        _, schema = fitted_pipeline_and_schema
        assert "INR" in schema["target_units"]
        assert schema["observation_year"] == config.OBSERVATION_YEAR

    def test_schema_brand_to_model_mapping_is_consistent(
        self, fitted_pipeline_and_schema
    ):
        """Dependent dropdowns rely on this mapping being correct."""
        _, schema = fitted_pipeline_and_schema
        for brand, models in schema["brand_to_models"].items():
            assert models, f"{brand} has no models"
            for model in models:
                assert model.startswith(brand)


class TestGroupSplitting:
    def test_grouped_split_keeps_a_vehicle_on_one_side(self):
        """The leakage guarantee the whole evaluation rests on."""
        frame = pd.DataFrame(
            {
                "name": ["Maruti Swift"] * 4 + [f"Car {i}" for i in range(40)],
                "year": [2014] * 4 + [2015] * 40,
                "km_driven": [50000] * 4 + list(range(10000, 50000, 1000)),
            }
        )
        groups = data_prep.vehicle_identity(frame)
        splitter = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        for train_idx, test_idx in splitter.split(frame, groups=groups):
            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])
            assert not (train_groups & test_groups)


class TestMetrics:
    def test_perfect_predictions_score_perfectly(self):
        y = np.array([100_000.0, 500_000.0, 1_000_000.0])
        metrics = evaluate.regression_metrics(y, y)
        assert metrics["mae_inr"] == 0
        assert metrics["rmse_inr"] == 0
        assert metrics["r2"] == pytest.approx(1.0)

    def test_metrics_are_in_rupees_not_log_units(self):
        """A 50,000-rupee miss must be reported as 50,000, not as log(50,000)."""
        y_true = np.array([400_000.0, 400_000.0])
        y_pred = np.array([450_000.0, 350_000.0])
        metrics = evaluate.regression_metrics(y_true, y_pred)
        assert metrics["mae_inr"] == pytest.approx(50_000)

    def test_rmse_exceeds_mae_when_errors_are_uneven(self):
        y_true = np.array([100_000.0] * 10)
        y_pred = np.array([100_000.0] * 9 + [1_000_000.0])
        metrics = evaluate.regression_metrics(y_true, y_pred)
        assert metrics["rmse_inr"] > metrics["mae_inr"]

    def test_price_band_breakdown_covers_the_test_rows(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(50_000, 3_000_000, 500)
        y_pred = y_true * rng.uniform(0.85, 1.15, 500)
        bands = evaluate.error_by_price_band(y_true, y_pred)
        assert bands["n"].sum() == len(y_true)
        assert (bands["mae_inr"] >= 0).all()

    def test_improvement_over_baseline(self):
        assert evaluate.improvement_over_baseline(50.0, 100.0) == pytest.approx(50.0)
        assert evaluate.improvement_over_baseline(100.0, 0.0) == 0.0


class TestBaseline:
    def test_baseline_predicts_the_training_median(self, trainable_frame):
        baseline = features.build_baseline()
        X = trainable_frame[config.FEATURES]
        y = trainable_frame[config.TARGET]
        baseline.fit(X, y)
        predictions = baseline.predict(X)
        assert len(set(np.round(predictions, 6))) == 1
        assert predictions[0] == pytest.approx(y.median())

    def test_every_registered_model_builds_and_fits(self, trainable_frame):
        X = trainable_frame[config.FEATURES]
        y = trainable_frame[config.TARGET]
        for name, builder in features.MODEL_BUILDERS.items():
            kwargs = {} if name in ("baseline_median", "ridge") else {"n_estimators": 10}
            pipeline = builder(**kwargs)
            pipeline.fit(X, y)
            predictions = pipeline.predict(X.head(3))
            assert np.isfinite(predictions).all(), f"{name} produced non-finite output"


def _example_row() -> dict:
    """One complete, plausible feature row."""
    return {
        "vehicle_age": 6,
        "km_driven": 60_000,
        "mileage_kmpl": 19.0,
        "engine_cc": 1200.0,
        "max_power_bhp": 90.0,
        "seats": 5.0,
        "brand": "Maruti",
        "model_family": "Maruti Swift",
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
    }
