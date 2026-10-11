# Used Car Price Predictor

Estimates the listing price of a used car in the Indian market from its brand,
model, age, odometer reading and specification — trained on ~6,900 CarDekho
listings, with a four-page Streamlit interface.

**Held-out test performance:** MAE **₹78,092** · RMSE **₹1,48,799** · R²
**0.928** · median error **12.0%** — a **73.8%** reduction in MAE against a
median-price baseline.

> This model reproduces **historical asking prices** (newest model year in the
> data: **2020**). It is not a current market valuation.

---

## Contents

- [What it does](#what-it-does)
- [Quickstart](#quickstart)
- [Results](#results)
- [Data](#data)
- [How it works](#how-it-works)
- [The three decisions that mattered](#the-three-decisions-that-mattered)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Licence](#licence)

---

## What it does

Four pages:

| Page | Purpose |
|---|---|
| **Predict Price** | Dependent brand → model dropdowns, a price estimate, and the model's measured error for that price band |
| **Data Explorer** | Filterable view of the training data, with the cleaning log |
| **Model Evaluation** | Baseline vs candidates, error by price band, permutation importance |
| **About** | Provenance, method, limitations |

---

## Quickstart

### Windows (PowerShell)

```powershell
git clone https://github.com/<your-username>/used-car-price-predictor.git
cd used-car-price-predictor

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts\download_data.py
python scripts\prepare_data.py
python scripts\train_model.py

python -m pytest
streamlit run app.py
```

> If PowerShell blocks activation, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first.

### macOS / Linux (bash)

```bash
git clone https://github.com/<your-username>/used-car-price-predictor.git
cd used-car-price-predictor

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train_model.py

python -m pytest
streamlit run app.py
```

The app opens at <http://localhost:8501>. Training takes about three minutes on
a laptop CPU; no GPU is used anywhere.

Both `data/raw/` and `models/` are committed, so **you can skip straight to
`streamlit run app.py`** after installing requirements.

---

## Results

All figures below are measured on **1,384 held-out listings** that no model saw
during cleaning, selection or tuning.

### Final model vs baseline

| Metric | Median baseline | Extra Trees | Improvement |
|---|---|---|---|
| MAE | ₹2,98,392 | **₹78,092** | **73.8% lower** |
| RMSE | ₹5,70,982 | **₹1,48,799** | 73.9% lower |
| R² | −0.061 | **0.928** | — |
| Median absolute % error | 42.9% | **12.0%** | — |

The baseline predicts the training median for every car. Its negative R² is not
a bug: the median minimises absolute error, not squared error, so it scores
marginally worse than predicting the mean. Quoting R² = 0.93 without this
comparison would hide how much of the work the baseline already does.

### Model comparison (grouped 5-fold CV, training data only)

| Model | CV MAE | CV RMSE | CV R² | Fit time |
|---|---|---|---|---|
| Median baseline | ₹2,72,796 | ₹5,22,547 | −0.047 | 0.3s |
| Ridge (log target) | ₹80,132 | ₹1,64,283 | 0.897 | 0.3s |
| Random Forest | ₹76,869 | ₹1,71,529 | 0.887 | 5.0s |
| **Extra Trees** | **₹75,518** | ₹1,67,330 | 0.893 | 6.4s |

Ridge finishing within ~6% of the ensembles is itself informative: most of the
price signal is close to log-linear in age and engine power.

### Error by price band

| Price band | Listings | MAE | Median % error | Mean bias |
|---|---|---|---|---|
| Under ₹2 lakh | 240 | ₹32,299 | 21.6% | **+₹16,439** |
| ₹2–4 lakh | 415 | ₹42,467 | 11.2% | +₹925 |
| ₹4–7 lakh | 433 | ₹66,448 | 10.0% | −₹1,308 |
| ₹7–12 lakh | 211 | ₹1,17,779 | 10.6% | −₹54,303 |
| Above ₹12 lakh | 85 | ₹3,42,127 | 14.1% | **−₹1,88,379** |

**This table is the most useful thing in the repository.** The model is
reliable in the ₹2–7 lakh range where the data is dense (61% of listings, ~10%
median error) and visibly regresses to the mean at both extremes: it
over-prices the cheapest cars by ₹16k on average and under-prices the most
expensive by ₹188k.

### What the model actually uses

Permutation importance on the test set — the rupees of MAE added by shuffling
one column:

| Feature | MAE increase |
|---|---|
| Vehicle age | ₹1,55,403 |
| Max power (bhp) | ₹94,186 |
| Engine (cc) | ₹43,581 |
| Model family | ₹27,556 |
| Transmission | ₹27,354 |
| Fuel | ₹25,335 |
| Brand | ₹21,209 |
| Owner | ₹7,338 |
| Kilometres driven | ₹7,258 |
| Seats | ₹1,860 |
| Seller type | ₹993 |
| Mileage (kmpl) | ₹116 |

Age and engine power dominate. **Odometer reading barely matters** (₹7,258)
once age is known — the two are correlated, and age carries the signal. That is
a genuinely counter-intuitive result and worth knowing before trusting a
kilometre-based valuation elsewhere.

Permutation importance is reported rather than the ensemble's built-in
`feature_importances_`, because impurity-based importance is biased towards
high-cardinality features, which would flatter `model_family` and its 210
categories.

---

## Data

- **Source:** [nehalbirla/vehicle-dataset-from-cardekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
- **File used:** `Car details v3.csv`
- **Licence:** Open Database License (ODbL) v1.0 — redistribution permitted
  with attribution, which is why `data/raw/` is committed here
- **Target units:** absolute Indian rupees
- **Observation period:** model years 1983–**2020**

### Why this file, and not the other three

The Kaggle download bundles four CSVs with three schemas and **two different
price units**:

| File | Rows | Price unit | Decision |
|---|---|---|---|
| **`Car details v3.csv`** | 8,128 | absolute INR | **used** — largest, and the only one with engine/power/mileage |
| `CAR DETAILS FROM CAR DEKHO.csv` | 4,340 | absolute INR | rejected — no specification columns |
| `car details v4.csv` | 2,059 | absolute INR | rejected — a quarter of the rows |
| `car data.csv` | 301 | **lakhs** | rejected — tiny, and `Present_Price` leaks the answer |

Reading `car data.csv` as rupees would understate every price by a factor of
100,000. The project carries an explicit `lakh_to_inr()` conversion, a schema
guard that rejects the wrong file, and tests for both.

### Cleaning and exclusions

Started from 8,128 raw rows, kept **6,924** (14.8% removed):

| Reason | Rows | Why |
|---|---|---|
| Exact duplicate rows | 1,202 | The scrape republished ~15% of listings verbatim. Removed **before** splitting. |
| Implied usage > 150,000 km/year | 2 | 410 km every day of the vehicle's life — physically impossible (one 2007 car listed at 2,360,457 km). |

**Nothing else was removed.** The ₹1-crore Volvo XC90, the 37-year-old cars and
the 500,000 km taxis all remain in training. Dropping awkward examples would
improve the headline metrics without improving the model.

Missing specification values (~3% for mileage, engine, power and seats) are
**imputed inside the pipeline**, not dropped — losing a whole listing over one
absent figure would waste real data. One `max_power` value in the raw file is
literally `" bhp"`; it parses to NaN and is imputed like any other gap.

The full log is written to `data/processed/exclusions.json`.

---

## How it works

```
raw CSV
  → parse "23.4 kmpl" / "1248 CC" / "74 bhp" into numbers
  → derive brand, model_family, vehicle_age (vs 2020)
  → drop duplicates and impossible records
  → GroupShuffleSplit on vehicle identity  ──────► test set (20%, untouched)
  → grouped 5-fold CV on the training set
      ├─ median baseline
      ├─ Ridge
      ├─ Random Forest
      └─ Extra Trees  ← selected
  → grid search (grouped CV, training only) + one-standard-error rule
  → fit on train, score the test set ONCE
  → save pipeline + feature schema + evaluation report
```

Preprocessing lives **inside** the scikit-learn `Pipeline`:

```python
ColumnTransformer(
    numeric:     SimpleImputer(median) [→ StandardScaler for Ridge]
    categorical: SimpleImputer(most_frequent)
                 → OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=15)
)
→ TransformedTargetRegressor(ExtraTrees, func=log1p, inverse=expm1)
```

Three things this buys:

- **No leakage, structurally.** Imputation medians and category lists can only
  be learned by calling `fit`, which only ever sees training folds.
- **Unseen categories cannot crash the app.** `infrequent_if_exist` routes any
  unknown brand into the "infrequent" bucket instead of raising — and the app
  tells the user it did.
- **A self-contained artefact.** The app loads one file and calls `predict` on
  raw columns.

**Why the log target.** Prices span ₹30,000 to ₹1 crore and are heavily
right-skewed. Fitted on raw rupees, squared-error models spend their capacity
on a handful of luxury cars. On the log scale the model optimises *relative*
error — what matters when quoting a price — and predictions can never come back
negative. Metrics are still reported in rupees.

---

## The three decisions that mattered

### 1. Vehicle age is measured against 2020, not the current year

The newest model year in the data is 2020. Computing age from `date.today()`
would add years of phantom age to every car and push every input beyond the
range the model was fitted on — producing confidently wrong answers rather than
an error. `config.OBSERVATION_YEAR` pins this, the prepare script warns if the
data disagrees with it, and a test asserts that age does not track today's year.

### 2. The split is grouped, not random

1,202 rows are byte-identical duplicates. After removing them, 348 vehicles
still appear more than once — the same car, year and odometer relisted at a
slightly different price (747 rows). A plain `train_test_split` would scatter
these across both sides and score the model partly on cars whose twin it had
memorised.

So the split is a `GroupShuffleSplit` keyed on `name + year + km_driven`, the
same grouping is used for every cross-validation fold, and the training script
asserts zero group overlap before continuing.

### 3. The simplest model that ties the best one wins

Grid search's top candidate (600 trees) beat 300 trees by **₹42** of CV MAE —
against a fold-to-fold standard error of **₹1,449**. That is noise, and taking
it would have cost a 61 MB artefact.

The one-standard-error rule keeps every candidate within one SE of the best and
picks the least complex. Result: **30.7 MB instead of 61.4 MB**, for ₹117 of
test MAE — about 0.15%.

---

## Project layout

```
used-car-price-predictor/
├── app.py                      # Streamlit entry point (navigation only)
├── app_pages/
│   ├── predict.py              # Predict Price
│   ├── explorer.py             # Data Explorer
│   ├── evaluation.py           # Model Evaluation
│   └── about.py                # About
├── src/ucpp/
│   ├── config.py               # paths, dataset facts, feature schema
│   ├── data_prep.py            # parsing, cleaning, unit conversion
│   ├── features.py             # pipeline builders
│   ├── evaluate.py             # metrics and breakdowns
│   ├── model_io.py             # save/load + schema
│   ├── predict.py              # input validation and warnings
│   ├── app_support.py          # cached loaders
│   └── console.py              # UTF-8 stdout on Windows
├── scripts/
│   ├── download_data.py        # Kaggle → data/raw/
│   ├── prepare_data.py         # → data/processed/cars_clean.csv
│   └── train_model.py          # → models/ + reports/
├── tests/                      # 80 tests
├── data/raw/                   # committed (ODbL permits redistribution)
├── models/                     # committed (the app needs them at runtime)
├── reports/evaluation.json     # every number in this README
├── requirements.txt
├── MODEL_CARD.md
└── README.md
```

**Training is completely separate from the app.** `app.py` never fits anything;
it loads artefacts produced by `scripts/train_model.py`. Start it without a
trained model and it tells you which commands to run.

---

## Testing

```powershell
python -m pytest              # all 80
python -m pytest -v           # verbose
python -m pytest -m artefacts # only the tests needing a trained model
```

**80 tests, all passing.** Coverage of the areas most likely to go wrong:

| Area | Examples |
|---|---|
| **Price-unit conversion** | lakh ↔ rupee round-trips; 3.35 lakh → ₹335,000; Indian digit grouping |
| **Schema validation** | rejects a frame missing columns; **rejects `car data.csv`** specifically |
| **Observation year** | a 2014 car is 6 years old; age does not track `date.today()`; age is never negative |
| **Cleaning** | duplicates removed; impossible odometers excluded; **expensive cars kept**; rows with unparseable specs kept and imputed |
| **Unknown categories** | an unseen value in *every* categorical column still predicts |
| **Model save/reload** | identical predictions after a disk round-trip; mismatched schema is rejected |
| **Group splitting** | no vehicle identity ever spans train and test |
| **End-to-end** | full prediction path; every brand in the schema predicts; **no fabricated confidence interval is exposed** |
| **App smoke tests** | all four pages render; the brand → model dropdown dependency populates; submitting shows a rupee price |

The app smoke tests use Streamlit's own `AppTest` harness and run the real
`app.py` in-process. They earned their keep: they caught that the input form
was wrapped in `st.form`, which suppresses reruns and would have left the
dependent Model dropdown **permanently empty** for every user.

Verified manually in a headless browser as well: 0 console errors, a 2015
Maruti Swift returning ₹5,36,348 with the depreciation chart rendering.

---

## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub (public, or private with Streamlit granted
   access).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **Create app → Deploy a public app from GitHub**, then set:
   - **Repository:** `<your-username>/used-car-price-predictor`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **Advanced settings → Python version: 3.12** *(see note)*
4. Deploy. The first build takes a few minutes while wheels install.

**Python version matters.** Community Cloud ignores `runtime.txt` and
`.python-version`; the only place to set it is the Advanced settings dialog.
The platform has been defaulting to newer releases than its docs state, and
there are open reports of 3.14 build failures with some package combinations.
**Pick 3.12** — every pin in `requirements.txt` has wheels there.

**Model artefacts must be in the repository.** Community Cloud installs
`requirements.txt` and runs `app.py`; it does **not** run the training script.
`models/price_pipeline.joblib` (30.7 MB) is committed for this reason — within
GitHub's 100 MB file limit and below the 50 MB warning threshold. To avoid
committing it, train on the host instead and uncomment the `models/` lines in
`.gitignore`.

### Security

- The app loads **only** `models/price_pipeline.joblib`, generated by this
  project's own training script. joblib files are pickles, and unpickling runs
  arbitrary code.
- **There is no model-upload control, by design.** Never add one, and never ask
  a user to upload a `.pkl`/`.joblib` file.
- No credentials, API keys or secrets are needed or stored anywhere.
  `.streamlit/secrets.toml` is gitignored.

---

## Limitations

- **Historical, not current.** Prices reflect listings up to model year 2020,
  with no inflation or market-shift adjustment.
- **Asking prices, not sale prices.** Sellers list above what they accept, so
  estimates skew high relative to real transactions.
- **No per-prediction confidence interval.** A tree-ensemble point prediction
  has no calibrated uncertainty. The app reports measured test-set error by
  price band instead of inventing a ± range — and a test enforces that no
  fabricated interval creeps in.
- **Regression to the mean at the extremes** — quantified in the table above.
- **Condition is invisible.** Accident history, service records and tyre/paint
  condition are absent and can move a real price more than the model's average
  error.
- **Trim is not modelled.** Predictions are at model-family level ("Maruti
  Swift"), and variants within a family differ substantially.
- **No location feature.** The chosen file has no city column, so regional
  price differences are not captured.
- **Indian market only.**

---

## Licence

Code: MIT. See [`LICENSE`](LICENSE).

Data: the CarDekho dataset is distributed under the
[Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/).
The copy in `data/raw/` is redistributed under those terms; attribution goes to
[Nehal Birla on Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho).

See [`MODEL_CARD.md`](MODEL_CARD.md) for intended use, evaluation detail and
ethical considerations.
