"""End-to-end tests: the full path a real prediction takes.

The tests marked ``artefacts`` exercise the genuinely trained model in
``models/``. They skip (rather than fail) when it has not been built yet, so a
fresh clone can still run the suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ucpp import app_support, config, data_prep, model_io, predict


def _example_values() -> dict:
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


class TestInputFrameConstruction:
    def test_builds_exactly_the_feature_columns_in_order(self):
        frame = predict.build_input_frame(_example_values())
        assert list(frame.columns) == config.FEATURES
        assert len(frame) == 1

    def test_extra_keys_are_ignored(self):
        values = _example_values() | {"colour": "red", "selling_price": 999}
        frame = predict.build_input_frame(values)
        assert list(frame.columns) == config.FEATURES

    def test_missing_keys_become_nan_rather_than_raising(self):
        values = _example_values()
        del values["mileage_kmpl"]
        frame = predict.build_input_frame(values)
        assert pd.isna(frame.at[0, "mileage_kmpl"])

    def test_numeric_strings_are_coerced(self):
        values = _example_values() | {"km_driven": "60000"}
        frame = predict.build_input_frame(values)
        assert frame.at[0, "km_driven"] == 60_000


class TestPredictionContract:
    def test_end_to_end_prediction_on_synthetic_model(self, fitted_pipeline_and_schema):
        """The core end-to-end path: values in, a sane rupee price out."""
        pipeline, schema = fitted_pipeline_and_schema
        result = predict.predict_price(pipeline, schema, _example_values())

        assert isinstance(result, predict.PredictionResult)
        assert result.price_inr > 0
        assert np.isfinite(result.price_inr)
        assert not result.is_extrapolation
        assert result.warnings == []

    def test_unseen_category_is_reported_not_hidden(self, fitted_pipeline_and_schema):
        pipeline, schema = fitted_pipeline_and_schema
        values = _example_values() | {"brand": "Koenigsegg"}
        result = predict.predict_price(pipeline, schema, values)

        assert result.price_inr > 0           # still answers
        assert "brand" in result.unseen_categories
        assert result.is_extrapolation
        assert any("Koenigsegg" in w for w in result.warnings)

    def test_out_of_range_numeric_is_reported(self, fitted_pipeline_and_schema):
        pipeline, schema = fitted_pipeline_and_schema
        values = _example_values() | {"km_driven": 900_000}
        result = predict.predict_price(pipeline, schema, values)

        assert "km_driven" in result.out_of_range
        assert result.out_of_range["km_driven"]["direction"] == "above"
        assert any("extrapolate" in w for w in result.warnings)

    def test_clean_input_produces_no_warnings(self, fitted_pipeline_and_schema):
        pipeline, schema = fitted_pipeline_and_schema
        result = predict.predict_price(pipeline, schema, _example_values())
        assert result.warnings == []

    def test_incomplete_input_raises_schema_error(self, fitted_pipeline_and_schema):
        pipeline, schema = fitted_pipeline_and_schema
        with pytest.raises(data_prep.SchemaError):
            # Bypass build_input_frame to simulate a malformed caller.
            predict.validate_feature_frame(pd.DataFrame([{"brand": "Maruti"}]))

    def test_no_confidence_interval_is_fabricated(self, fitted_pipeline_and_schema):
        """The result object must not expose an invented uncertainty range.

        Guards against a well-meaning future edit adding a '+/- 10%' band that
        has no statistical basis.
        """
        pipeline, schema = fitted_pipeline_and_schema
        result = predict.predict_price(pipeline, schema, _example_values())
        for forbidden in ("confidence", "interval", "ci_low", "ci_high", "std", "margin"):
            assert not hasattr(result, forbidden), (
                f"PredictionResult exposes '{forbidden}' - a tree-ensemble point "
                f"prediction has no calibrated uncertainty to report"
            )

    def test_older_cars_predict_lower_prices(self, fitted_pipeline_and_schema):
        """A directional sanity check: depreciation must go the right way."""
        pipeline, schema = fitted_pipeline_and_schema
        young = predict.predict_price(
            pipeline, schema, _example_values() | {"vehicle_age": 1}
        )
        old = predict.predict_price(
            pipeline, schema, _example_values() | {"vehicle_age": 12}
        )
        assert young.price_inr > old.price_inr


# --------------------------------------------------------------------------
# Tests against the genuinely trained artefacts.
# --------------------------------------------------------------------------
requires_artefacts = pytest.mark.skipif(
    not app_support.artefacts_available(),
    reason="trained model not present; run python scripts/train_model.py",
)


@pytest.mark.artefacts
@requires_artefacts
class TestRealArtefacts:
    def test_real_model_loads(self):
        pipeline, schema = model_io.load_model()
        assert schema["features"] == config.FEATURES
        assert schema["observation_year"] == config.OBSERVATION_YEAR

    def test_real_model_predicts_a_plausible_price(self):
        """A 2014 Swift with 60,000 km should land in a sane rupee range."""
        pipeline, schema = model_io.load_model()
        result = predict.predict_price(pipeline, schema, _example_values())
        assert 100_000 < result.price_inr < 2_000_000, (
            f"got {result.price_inr:,.0f} for a mid-age Maruti Swift"
        )

    def test_real_model_handles_an_unseen_brand(self):
        pipeline, schema = model_io.load_model()
        values = _example_values() | {
            "brand": "Koenigsegg",
            "model_family": "Koenigsegg Jesko",
        }
        result = predict.predict_price(pipeline, schema, values)
        assert result.price_inr > 0
        assert result.is_extrapolation

    def test_every_brand_in_the_schema_predicts(self):
        """Exercises every dropdown option the app can offer."""
        pipeline, schema = model_io.load_model()
        for brand, models in schema["brand_to_models"].items():
            values = _example_values() | {"brand": brand, "model_family": models[0]}
            result = predict.predict_price(pipeline, schema, values)
            assert result.price_inr > 0, f"{brand} produced {result.price_inr}"

    def test_saved_metrics_are_present_and_sane(self):
        report = app_support.get_evaluation_report()
        assert report is not None
        test_metrics = report["test_metrics"]

        assert test_metrics["mae_inr"] > 0
        assert 0 < test_metrics["r2"] < 1
        # RMSE must be at least MAE by definition.
        assert test_metrics["rmse_inr"] >= test_metrics["mae_inr"]
        # The model must beat the median baseline, or it has learned nothing.
        assert test_metrics["mae_inr"] < report["baseline_test_metrics"]["mae_inr"]
        assert report["mae_improvement_over_baseline_pct"] > 0

    def test_reported_split_sizes_add_up(self):
        report = app_support.get_evaluation_report()
        dataset = report["dataset"]
        assert dataset["train_rows"] + dataset["test_rows"] == dataset["clean_rows"]

    def test_clean_data_loads_without_the_processed_file(self, monkeypatch, tmp_path):
        """The app must work on a fresh clone, where data/processed/ is absent.

        Regression test: data/processed/ is gitignored, so on a fresh clone -
        and on Streamlit Community Cloud, which never runs the training
        scripts - the Data Explorer used to crash with FileNotFoundError. The
        loader now rebuilds the table from the committed raw CSV.
        """
        monkeypatch.setattr(config, "CLEAN_CSV_PATH", tmp_path / "absent.csv")
        app_support.get_clean_data.clear()
        try:
            frame = app_support.get_clean_data()
            assert len(frame) > 0
            for feature in config.FEATURES:
                assert feature in frame.columns
        finally:
            app_support.get_clean_data.clear()

    def test_exclusion_log_available_without_the_processed_file(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(config, "EXCLUSION_LOG_PATH", tmp_path / "absent.json")
        app_support.get_exclusion_log.clear()
        try:
            log = app_support.get_exclusion_log()
            assert log is not None
            assert log["rows_in"] > log["rows_out"] > 0
        finally:
            app_support.get_exclusion_log.clear()

    def test_round_trip_through_disk_is_stable(self, tmp_path):
        """Save/reload of the real artefact must not change predictions."""
        pipeline, schema = model_io.load_model()
        sample = predict.build_input_frame(_example_values())
        before = pipeline.predict(sample)

        model_path = tmp_path / "copy.joblib"
        schema_path = tmp_path / "copy.json"
        model_io.save_model(pipeline, schema, model_path, schema_path)
        reloaded, _ = model_io.load_model(model_path, schema_path)

        np.testing.assert_array_almost_equal(before, reloaded.predict(sample))
