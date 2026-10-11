"""Tests for schema validation, unit conversion and cleaning rules."""

from __future__ import annotations

import pandas as pd
import pytest

from ucpp import config, data_prep


# ---------------------------------------------------------------- unit conversion
class TestPriceUnits:
    """The lakh/rupee distinction is the single most dangerous trap in this
    dataset: 'car data.csv' quotes lakhs while the file we train on quotes
    rupees, and confusing them is a factor-of-100,000 error."""

    def test_lakh_to_inr_converts_by_one_hundred_thousand(self):
        assert data_prep.lakh_to_inr(1) == 100_000
        assert data_prep.lakh_to_inr(3.35) == 335_000
        assert data_prep.lakh_to_inr(35) == 3_500_000

    def test_inr_to_lakh_is_the_inverse(self):
        assert data_prep.inr_to_lakh(450_000) == 4.5
        for value in (0.1, 3.35, 12.0, 35.0):
            assert data_prep.lakh_to_inr(data_prep.inr_to_lakh(
                data_prep.lakh_to_inr(value))) == pytest.approx(
                data_prep.lakh_to_inr(value))

    def test_round_trip_preserves_value(self):
        for rupees in (29_999, 450_000, 10_000_000):
            assert data_prep.lakh_to_inr(data_prep.inr_to_lakh(rupees)) == pytest.approx(
                rupees
            )

    def test_realistic_lakh_prices_land_in_the_rupee_range(self):
        """A 3.35-lakh car must become 335,000 rupees, not 3.35 rupees."""
        converted = data_prep.lakh_to_inr(3.35)
        assert config.MIN_PLAUSIBLE_PRICE_INR <= converted <= config.MAX_PLAUSIBLE_PRICE_INR

    def test_rupees_per_lakh_constant(self):
        assert config.RUPEES_PER_LAKH == 100_000


class TestIndianFormatting:
    def test_groups_digits_in_the_indian_convention(self):
        assert data_prep.format_inr(450000) == "₹4,50,000"
        assert data_prep.format_inr(10000000) == "₹1,00,00,000"
        assert data_prep.format_inr(999) == "₹999"
        assert data_prep.format_inr(78092) == "₹78,092"

    def test_handles_negative_values(self):
        assert data_prep.format_inr(-54303).startswith("-₹")


# ------------------------------------------------------------ schema validation
class TestSchemaValidation:
    def test_accepts_the_real_raw_schema(self, raw_rows):
        data_prep.validate_raw_schema(raw_rows)  # must not raise

    def test_rejects_a_frame_missing_columns(self, raw_rows):
        broken = raw_rows.drop(columns=["max_power"])
        with pytest.raises(data_prep.SchemaError, match="max_power"):
            data_prep.validate_raw_schema(broken)

    def test_rejects_the_wrong_file_from_the_same_download(self):
        """'car data.csv' has title-cased columns AND prices in lakhs.

        This is the guard that stops the wrong file from the same Kaggle zip
        silently training a model on values 100,000x too small.
        """
        wrong_file = pd.DataFrame(
            {
                "Car_Name": ["ritz"], "Year": [2014], "Selling_Price": [3.35],
                "Present_Price": [5.59], "Kms_Driven": [27000],
                "Fuel_Type": ["Petrol"], "Seller_Type": ["Dealer"],
                "Transmission": ["Manual"], "Owner": [0],
            }
        )
        with pytest.raises(data_prep.SchemaError):
            data_prep.validate_raw_schema(wrong_file)

    def test_feature_frame_validation_rejects_missing_features(self):
        with pytest.raises(data_prep.SchemaError, match="vehicle_age"):
            data_prep.validate_feature_frame(pd.DataFrame({"brand": ["Maruti"]}))

    def test_feature_frame_validation_accepts_a_complete_row(self):
        complete = pd.DataFrame([{feature: 1 for feature in config.FEATURES}])
        data_prep.validate_feature_frame(complete)  # must not raise


# ----------------------------------------------------------------- text parsing
class TestNumberExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("23.4 kmpl", 23.4),
            ("1248 CC", 1248.0),
            ("74 bhp", 74.0),
            ("261.49 bhp", 261.49),
            ("16.1 km/kg", 16.1),
        ],
    )
    def test_parses_unit_suffixed_numbers(self, text, expected):
        assert data_prep._extract_number(text) == pytest.approx(expected)

    def test_returns_nan_for_values_with_no_digits(self):
        """' bhp' appears verbatim in the real file and must not crash."""
        assert pd.isna(data_prep._extract_number(" bhp"))
        assert pd.isna(data_prep._extract_number(None))
        assert pd.isna(data_prep._extract_number(float("nan")))


# --------------------------------------------------------------------- cleaning
class TestCleaning:
    def test_age_is_measured_against_the_observation_year(self, raw_rows):
        """The central correctness check of the whole project.

        A 2014 car in a dataset observed in 2020 is 6 years old, regardless of
        what year it is when the code runs.
        """
        derived = data_prep.derive_features(raw_rows)
        swift = derived[derived["year"] == 2014].iloc[0]
        assert swift["vehicle_age"] == config.OBSERVATION_YEAR - 2014 == 6

    def test_age_does_not_depend_on_the_current_year(self, raw_rows):
        import datetime

        derived = data_prep.derive_features(raw_rows)
        max_age = derived["vehicle_age"].max()
        naive_max_age = datetime.date.today().year - raw_rows["year"].min()
        assert max_age < naive_max_age, (
            "vehicle_age appears to be computed from today's year, which would "
            "push every input outside the model's fitted range"
        )

    def test_newest_cars_have_age_zero_not_negative(self, raw_rows):
        derived = data_prep.derive_features(raw_rows)
        assert derived["vehicle_age"].min() >= 0

    def test_brand_and_model_family_are_derived_from_the_name(self, raw_rows):
        derived = data_prep.derive_features(raw_rows)
        row = derived.iloc[0]
        assert row["brand"] == "Maruti"
        assert row["model_family"] == "Maruti Swift"

    def test_exact_duplicates_are_removed(self, raw_rows):
        clean, log = data_prep.clean(raw_rows)
        duplicate_step = next(
            s for s in log.steps if s["reason"] == "exact_duplicate_rows"
        )
        assert duplicate_step["rows_removed"] == 1

    def test_impossible_odometer_is_excluded(self, raw_rows):
        clean, log = data_prep.clean(raw_rows)
        step = next(s for s in log.steps if s["reason"] == "impossible_mileage_rate")
        assert step["rows_removed"] == 1
        assert 2360457 not in clean["km_driven"].values

    def test_expensive_cars_are_kept(self, raw_rows):
        """Dropping the luxury tail would flatter the metrics dishonestly."""
        clean, _ = data_prep.clean(raw_rows)
        assert 7_200_000 in clean[config.TARGET].values

    def test_rows_with_unparseable_specs_are_kept_not_dropped(self, raw_rows):
        """A missing power figure must not cost us the whole listing."""
        clean, _ = data_prep.clean(raw_rows)
        nano = clean[clean["name"].str.contains("Nano")]
        assert len(nano) == 1
        assert pd.isna(nano.iloc[0]["max_power_bhp"])

    def test_clean_output_has_exactly_the_model_features(self, raw_rows):
        clean, _ = data_prep.clean(raw_rows)
        for feature in config.FEATURES:
            assert feature in clean.columns
        assert config.TARGET in clean.columns

    def test_exclusion_log_accounts_for_every_row(self, raw_rows):
        clean, log = data_prep.clean(raw_rows)
        assert log.rows_in == len(raw_rows)
        assert log.rows_out == len(clean)
        assert sum(s["rows_removed"] for s in log.steps) == log.rows_in - log.rows_out

    def test_cleaning_does_not_mutate_the_input(self, raw_rows):
        before = raw_rows.copy()
        data_prep.clean(raw_rows)
        pd.testing.assert_frame_equal(raw_rows, before)


class TestVehicleIdentity:
    def test_same_vehicle_gets_the_same_key(self, raw_rows):
        identity = data_prep.vehicle_identity(raw_rows)
        # Rows 0 and 3 are the same listing.
        assert identity.iloc[0] == identity.iloc[3]

    def test_different_vehicles_get_different_keys(self, raw_rows):
        identity = data_prep.vehicle_identity(raw_rows)
        assert identity.iloc[0] != identity.iloc[1]

    def test_key_is_case_insensitive_on_name(self):
        frame = pd.DataFrame(
            {
                "name": ["Maruti Swift", "maruti swift"],
                "year": [2014, 2014],
                "km_driven": [50000, 50000],
            }
        )
        identity = data_prep.vehicle_identity(frame)
        assert identity.iloc[0] == identity.iloc[1]
