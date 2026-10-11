"""About page: provenance, method, limitations."""

from __future__ import annotations

import streamlit as st

from ucpp import app_support as sup
from ucpp import config


def render() -> None:
    st.header("About this project")

    report = sup.get_evaluation_report()

    st.markdown(
        f"""
A used-car price estimator built end to end as a machine-learning portfolio
project: data preparation, model comparison, honest evaluation, and this app.

**What it does.** Given a car's brand, model family, age, odometer reading,
fuel, transmission, ownership history and basic engine specification, it
estimates the rupee price the vehicle would have been *listed* at in the
Indian used-car market of the dataset's period.

**What it is not.** It is not a current market valuation, not a sale-price
guarantee, and not a substitute for an inspection.
"""
    )

    st.divider()
    st.subheader("Data")

    st.markdown(
        f"""
- **Source:** [{config.KAGGLE_DATASET_REF}]({config.KAGGLE_DATASET_URL}) on Kaggle
- **File used:** `{config.RAW_CSV_NAME}` (one of four files in that download)
- **Licence:** {config.DATASET_LICENCE}
- **Target units:** absolute Indian rupees (INR)
- **Observation period:** model years up to **{config.OBSERVATION_YEAR}**

**Why that file, and not the other three.** The download bundles four CSVs with
three different schemas and *two different price units*:

| File | Rows | Price unit | Verdict |
|---|---|---|---|
| `Car details v3.csv` | 8,128 | absolute INR | **used** — largest, richest features |
| `CAR DETAILS FROM CAR DEKHO.csv` | 4,340 | absolute INR | rejected — no engine/power columns |
| `car details v4.csv` | 2,059 | absolute INR | rejected — only a quarter the rows |
| `car data.csv` | 301 | **lakhs** | rejected — tiny, and leaks `Present_Price` |

Treating `car data.csv` as rupees would understate prices by a factor of
100,000. The project includes an explicit `lakh_to_inr` conversion and a test
for it precisely because this trap is one line of careless code away.
"""
    )

    st.divider()
    st.subheader("Method, and the three decisions that mattered")

    st.markdown(
        f"""
**1. Age is measured against {config.OBSERVATION_YEAR}, not the current year.**
The newest model year in the data is {config.OBSERVATION_YEAR}. Computing age
from today's calendar year would add years of phantom age to every car and push
every input past the range the model was fitted on — a mistake that produces
confidently wrong answers rather than an error.

**2. The split is grouped, not random.** The raw file contains 1,202 exact
duplicate rows (14.8%), plus several hundred near-duplicates: the same car,
year and odometer relisted at a slightly different price. Those duplicates were
removed before splitting, and the train/test split is then grouped on vehicle
identity so no vehicle appears on both sides. A plain random split would score
the model partly on cars whose twin it had memorised.

**3. Hard examples were kept.** Only duplicated and *physically impossible*
records were excluded — 1,202 duplicates and 2 listings implying over
150,000 km per year. The ₹1-crore Volvo, the 37-year-old cars and the
500,000 km taxis all remain in the training data. Dropping them would improve
the headline metrics without improving the model.

Preprocessing (median imputation, one-hot encoding) lives *inside* the
scikit-learn pipeline, so it is only ever fitted on training folds. The target
is modelled as `log1p(price)` and inverted for reporting, because prices span
₹30,000 to ₹1 crore and squared error on raw rupees would chase the luxury tail.
"""
    )

    if report:
        test = report["test_metrics"]
        st.divider()
        st.subheader("Verified results")
        st.markdown(
            f"""
Measured on **{report['dataset']['test_rows']:,} held-out listings**, scored
once after model selection and tuning were complete:

| Metric | Median baseline | {report['selected_model']} |
|---|---|---|
| MAE | {sup.format_inr(report['baseline_test_metrics']['mae_inr'])} | **{sup.format_inr(test['mae_inr'])}** |
| RMSE | {sup.format_inr(report['baseline_test_metrics']['rmse_inr'])} | **{sup.format_inr(test['rmse_inr'])}** |
| R² | {report['baseline_test_metrics']['r2']:.4f} | **{test['r2']:.4f}** |
| Median abs % error | {report['baseline_test_metrics']['median_abs_pct_error']:.1f}% | **{test['median_abs_pct_error']:.1f}%** |

See the **Model Evaluation** page for the breakdown by price band.
"""
        )

    st.divider()
    st.subheader("Limitations")

    st.markdown(
        """
- **Historical, not current.** Prices reflect the dataset's period, not today's
  market. No inflation or market-shift adjustment is applied.
- **Asking prices, not transaction prices.** Sellers list above what they accept.
- **No uncertainty estimate per prediction.** A tree-ensemble point prediction
  carries no calibrated confidence interval. The app reports measured test-set
  error by price band instead of inventing a ± range.
- **Regression to the mean at the extremes.** The model over-predicts the
  cheapest cars and under-predicts the most expensive ones. Quantified on the
  Model Evaluation page.
- **Condition is invisible.** Accident history, service records and location are
  not in the data and can move a real price more than the model's average error.
- **Trim is not modelled.** Predictions are at model-family level; variant
  differences within a family can be large.
- **Geographic blind spot.** The chosen file has no location column, so regional
  price differences are not captured.
"""
    )

    st.divider()
    st.subheader("Stack and reproducibility")

    schema_env = {}
    if sup.artefacts_available():
        _, schema = sup.get_model()
        schema_env = schema.get("environment", {})

    st.markdown(
        f"""
Python, pandas, scikit-learn, Streamlit, Plotly and joblib. CPU only — training
takes a couple of minutes on a laptop.

{"**Model trained with:** " + ", ".join(f"{k} {v}" for k, v in schema_env.items()) if schema_env else ""}

Reproduce from scratch:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train_model.py
streamlit run app.py
```

**On model files.** Training is fully separate from app startup. The app loads
only the artefact produced by this project's own training script, from this
project's `models/` directory. It has no upload control: joblib files are
pickles, and loading an untrusted one executes arbitrary code.
"""
    )
