"""Shared fixtures.

The synthetic frame here is deliberately small and hand-built: the unit tests
must pass without the Kaggle download present, so CI and a fresh clone can run
them immediately. Tests that genuinely need the real artefacts are marked and
skip cleanly when those are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ucpp import config  # noqa: E402


@pytest.fixture
def raw_rows() -> pd.DataFrame:
    """A tiny frame with the real raw schema, including the known nasty cases.

    Row 3 duplicates row 0 exactly. Row 4 has an unparseable max_power (' bhp'
    appears verbatim in the real file) and a missing seat count. Row 5 has a
    physically impossible odometer reading.
    """
    rows = [
        {
            "name": "Maruti Swift Dzire VDI", "year": 2014, "selling_price": 450000,
            "km_driven": 145500, "fuel": "Diesel", "seller_type": "Individual",
            "transmission": "Manual", "owner": "First Owner", "mileage": "23.4 kmpl",
            "engine": "1248 CC", "max_power": "74 bhp",
            "torque": "190Nm@ 2000rpm", "seats": 5.0,
        },
        {
            "name": "Hyundai i20 Asta 1.2", "year": 2018, "selling_price": 650000,
            "km_driven": 25000, "fuel": "Petrol", "seller_type": "Dealer",
            "transmission": "Manual", "owner": "First Owner", "mileage": "18.6 kmpl",
            "engine": "1197 CC", "max_power": "82 bhp",
            "torque": "115Nm@ 4000rpm", "seats": 5.0,
        },
        {
            "name": "BMW X7 xDrive 30d DPE", "year": 2020, "selling_price": 7200000,
            "km_driven": 5000, "fuel": "Diesel", "seller_type": "Dealer",
            "transmission": "Automatic", "owner": "First Owner",
            "mileage": "13.38 kmpl", "engine": "2993 CC", "max_power": "261.49 bhp",
            "torque": "620Nm@ 2000rpm", "seats": 7.0,
        },
        {  # exact duplicate of row 0
            "name": "Maruti Swift Dzire VDI", "year": 2014, "selling_price": 450000,
            "km_driven": 145500, "fuel": "Diesel", "seller_type": "Individual",
            "transmission": "Manual", "owner": "First Owner", "mileage": "23.4 kmpl",
            "engine": "1248 CC", "max_power": "74 bhp",
            "torque": "190Nm@ 2000rpm", "seats": 5.0,
        },
        {  # unparseable max_power and missing seats, as in the real file
            "name": "Tata Nano Cx BSIV", "year": 2012, "selling_price": 90000,
            "km_driven": 40000, "fuel": "Petrol", "seller_type": "Individual",
            "transmission": "Manual", "owner": "Second Owner", "mileage": "25.4 kmpl",
            "engine": "624 CC", "max_power": " bhp",
            "torque": "51Nm@ 4000rpm", "seats": None,
        },
        {  # physically impossible: 2.36M km on a 2007 car
            "name": "Hyundai i20 Magna", "year": 2007, "selling_price": 550000,
            "km_driven": 2360457, "fuel": "Petrol", "seller_type": "Individual",
            "transmission": "Manual", "owner": "Third Owner", "mileage": "17.0 kmpl",
            "engine": "1197 CC", "max_power": "80 bhp",
            "torque": "112Nm@ 4000rpm", "seats": 5.0,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def trainable_frame() -> pd.DataFrame:
    """A frame large and varied enough to actually fit a pipeline on.

    Prices are generated from a deterministic formula so the tests assert on
    pipeline mechanics (fitting, saving, reloading, unseen categories) without
    depending on the real dataset being downloaded.
    """
    rows = []
    brands = [
        ("Maruti", "Maruti Swift", 90.0, 1200.0),
        ("Hyundai", "Hyundai i20", 85.0, 1197.0),
        ("Honda", "Honda City", 115.0, 1500.0),
        ("Toyota", "Toyota Innova", 150.0, 2400.0),
    ]
    for brand, family, power, cc in brands:
        for age in range(0, 15):
            for fuel in ("Petrol", "Diesel"):
                for transmission in ("Manual", "Automatic"):
                    price = max(
                        40000.0,
                        900000.0 * (0.88**age) * (1.25 if transmission == "Automatic" else 1.0),
                    )
                    rows.append(
                        {
                            "name": f"{family} Variant",
                            "year": config.OBSERVATION_YEAR - age,
                            "vehicle_age": age,
                            "km_driven": 10000 + age * 9000,
                            "mileage_kmpl": 19.0,
                            "engine_cc": cc,
                            "max_power_bhp": power,
                            "seats": 5.0,
                            "brand": brand,
                            "model_family": family,
                            "fuel": fuel,
                            "seller_type": "Individual",
                            "transmission": transmission,
                            "owner": "First Owner",
                            config.TARGET: price,
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_pipeline_and_schema(trainable_frame):
    """A real fitted pipeline plus its schema, built from synthetic data."""
    from ucpp import features, model_io

    pipeline = features.build_extra_trees(n_estimators=25)
    pipeline.fit(trainable_frame[config.FEATURES], trainable_frame[config.TARGET])
    schema = model_io.build_schema(trainable_frame, "extra_trees_test")
    return pipeline, schema
