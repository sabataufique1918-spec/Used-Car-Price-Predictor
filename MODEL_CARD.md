# Model Card — Used Car Price Predictor

## Model details

| | |
|---|---|
| **Model** | `ExtraTreesRegressor` (300 trees, `min_samples_leaf=2`, `max_features=1.0`) inside a scikit-learn `Pipeline` |
| **Target** | `log1p(selling_price)`, inverted to absolute INR for all reporting |
| **Task** | Supervised regression (tabular) |
| **Version** | 1.0.0 |
| **Artefact** | `models/price_pipeline.joblib` (30.7 MB) + `models/feature_schema.json` |
| **Framework** | scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.2, Python 3.14 |
| **Compute** | CPU only; ~3 minutes end to end on a laptop |
| **Licence** | MIT (code); data ODbL v1.0 |

The saved artefact is a complete pipeline: imputation, one-hot encoding, the
target transform and the estimator. It accepts raw feature columns and returns
rupees.

### Inputs

| Feature | Type | Notes |
|---|---|---|
| `vehicle_age` | numeric | **Years relative to 2020**, not to the current year |
| `km_driven` | numeric | Odometer reading |
| `mileage_kmpl` | numeric | Parsed from `"23.4 kmpl"` |
| `engine_cc` | numeric | Parsed from `"1248 CC"` |
| `max_power_bhp` | numeric | Parsed from `"74 bhp"` |
| `seats` | numeric | |
| `brand` | categorical | First token of the listing title (32 values) |
| `model_family` | categorical | First two tokens (210 values) |
| `fuel` | categorical | Petrol / Diesel / CNG / LPG |
| `seller_type` | categorical | Individual / Dealer / Trustmark Dealer |
| `transmission` | categorical | Manual / Automatic |
| `owner` | categorical | First … Fourth & Above Owner, Test Drive Car |

**Deliberately excluded:** `torque` — its raw values mix units and rev ranges
(`"190Nm@ 2000rpm"`, `"22.4 kgm at 1750-2750rpm"`), so parsing it into one
comparable number is guesswork, and it largely duplicates `engine_cc` and
`max_power_bhp`.

Missing numeric inputs are imputed with the **training-fold** median; missing
categoricals with the training-fold mode. Categories seen fewer than 15 times
in training, **and any category never seen at all**, are routed to a shared
"infrequent" bucket (`handle_unknown="infrequent_if_exist"`) rather than
raising.

### Output

A single price in absolute Indian rupees. **No confidence interval**, by
design — see [Limitations](#limitations).

---

## Intended use

**Intended.** A portfolio and teaching artefact demonstrating an end-to-end
tabular ML workflow: leakage-aware splitting, pipeline-contained preprocessing,
baseline comparison and honest error reporting. Also a rough historical
reference point for what a given used car was *listed* at in the Indian market
of the dataset's period.

**Not intended.** Any of the following would be a misuse:

- Current market valuation, or pricing a real sale, trade-in or insurance claim
- Loan collateral assessment or any financial underwriting
- Markets outside India, or vehicle types outside the training distribution
  (commercial vehicles, imports, modified cars)
- Any decision where being wrong by the error margins below would matter, taken
  without a human check

**Users.** Learners reading the code; recruiters assessing it; anyone curious
about historical Indian used-car prices.

---

## Training data

- **Source:** [nehalbirla/vehicle-dataset-from-cardekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho), file `Car details v3.csv`
- **Licence:** Open Database License (ODbL) v1.0
- **Raw size:** 8,128 listings; **after cleaning:** 6,924
- **Split:** 5,540 train / 1,384 test (20%), `GroupShuffleSplit` on vehicle identity
- **Period:** model years 1983–2020 (observation year **2020**)
- **Price range:** ₹29,999 – ₹1,00,00,000 (median ₹4,00,000)
- **Target units:** absolute INR (a sibling file in the same download quotes
  lakhs; it is not used)

### Exclusions

| Reason | Rows | Rationale |
|---|---|---|
| Exact duplicate rows | 1,202 | Scrape artefacts; removed **before** splitting to prevent memorisation across train/test |
| Implied usage > 150,000 km/year | 2 | ~410 km every day of the vehicle's life; physically impossible |

**14.8% removed in total, and nothing was excluded for being inconvenient.**
High-price, high-mileage and very old vehicles were all retained. Removing hard
examples would improve the reported metrics without improving the model, which
would make this card dishonest.

### Known data characteristics

- **Heavily imbalanced by brand.** Maruti (2,448) and Hyundai (1,415) are over
  half the data; BMW (120) and Skoda (105) are thin. Predictions for sparse
  brands rest on far less evidence.
- **Right-skewed prices.** The median is ₹4 lakh but the maximum is ₹1 crore.
- **Seller mix.** 78% individual sellers, which shapes the price distribution.
- **~3% missing specifications**, imputed rather than dropped.
- **No location column**, so regional price variation is invisible to the model.

---

## Evaluation

Measured on **1,384 held-out listings**, scored **once** after model selection
and hyper-parameter tuning were complete. The test set was not used for
cleaning decisions, model choice or tuning.

### Headline

| Metric | Median baseline | Extra Trees |
|---|---|---|
| MAE | ₹2,98,392 | **₹78,092** |
| RMSE | ₹5,70,982 | **₹1,48,799** |
| R² | −0.061 | **0.928** |
| Median absolute % error | 42.9% | **12.0%** |

MAE is **73.8% lower** than the baseline.

The baseline's negative R² is expected: predicting the median minimises
absolute error, not squared error, so it scores slightly below predicting the
mean. It is included because an R² of 0.93 means little without knowing what a
trivial model achieves.

### Disaggregated by price band

| Price band | n | MAE | RMSE | Median % error | Mean bias |
|---|---|---|---|---|---|
| Under ₹2 lakh | 240 | ₹32,299 | ₹42,614 | 21.6% | **+₹16,439** |
| ₹2–4 lakh | 415 | ₹42,467 | ₹56,536 | 11.2% | +₹925 |
| ₹4–7 lakh | 433 | ₹66,448 | ₹97,692 | 10.0% | −₹1,308 |
| ₹7–12 lakh | 211 | ₹1,17,779 | ₹1,62,316 | 10.6% | −₹54,303 |
| Above ₹12 lakh | 85 | ₹3,42,127 | ₹4,75,142 | 14.1% | **−₹1,88,379** |

**The disaggregated view is the one that matters.** Performance is not uniform:
the model is dependable in the ₹2–7 lakh range (61% of listings, ~10% median
error) and systematically biased at both ends — over-pricing cheap cars by
₹16k on average, under-pricing expensive ones by ₹188k. This is classic
regression to the mean and it is a property of the model, not noise.

RMSE exceeding MAE by roughly 1.9× indicates a meaningful tail of large misses
rather than uniformly moderate error.

### Model selection

Grouped 5-fold CV on the training set only:

| Model | CV MAE | CV RMSE | CV R² |
|---|---|---|---|
| Median baseline | ₹2,72,796 | ₹5,22,547 | −0.047 |
| Ridge (log target) | ₹80,132 | ₹1,64,283 | 0.897 |
| Random Forest | ₹76,869 | ₹1,71,529 | 0.887 |
| **Extra Trees** | **₹75,518** | ₹1,67,330 | 0.893 |

Hyper-parameters were chosen by grid search under the same grouped CV, then
narrowed with the **one-standard-error rule**: the top candidate (600 trees)
beat 300 trees by ₹42 against a standard error of ₹1,449, so the simpler model
was taken. Cost: ₹117 of test MAE (~0.15%). Benefit: a 30.7 MB artefact instead
of 61.4 MB.

### Feature importance (permutation, test set)

Vehicle age (₹1,55,403) and max power (₹94,186) dominate; engine size
(₹43,581) follows. Notably, **kilometres driven contributes only ₹7,258** once
age is known, and mileage (kmpl) is effectively unused (₹116).

Permutation importance is used instead of impurity-based
`feature_importances_`, which is biased towards high-cardinality features and
would overstate `model_family`.

---

## Limitations

1. **Historical, not current.** Trained on listings up to model year 2020, with
   no adjustment for inflation or market change. It is not a present-day
   valuation and has not been validated against current prices.
2. **Asking prices, not transaction prices.** The dataset records what sellers
   listed at, typically above what they accepted. Estimates skew high relative
   to real sale values.
3. **No per-prediction uncertainty.** A tree-ensemble point prediction carries
   no calibrated confidence interval. Quoting "±10%" or a per-prediction
   accuracy percentage would be fabricating a number. The application shows
   measured test-set error for the relevant price band instead, and an
   automated test fails the build if a confidence-interval field is ever added
   to the prediction result.
4. **Systematic bias at the price extremes**, quantified above.
5. **Condition is unobserved.** Accident history, service records, tyres, paint
   and mechanical state are absent and can move a real price by more than the
   model's average error.
6. **Trim is not modelled.** `model_family` is two tokens ("Maruti Swift"), so
   a base and top variant of the same model are indistinguishable.
7. **Geography is unobserved.** The chosen file has no city column.
8. **Thin coverage for rare brands.** Luxury and low-volume marques have few
   examples; the ₹12 lakh+ band is 6% of the test set.
9. **India only.** Nothing here transfers to other markets.
10. **Distribution shift is unmonitored.** There is no drift detection; the
    model will not know when it has become stale, because it already is.

---

## Ethical considerations

- **Information asymmetry.** A price estimate can disadvantage whichever party
  trusts it more. The app states prominently and on every page that figures are
  historical listing prices, not valuations.
- **Fairness.** The model uses no personal or demographic attributes — only
  vehicle characteristics. But `seller_type` encodes a market structure
  (individual vs dealer) that correlates with bargaining power, and the data's
  brand imbalance means owners of less common vehicles get less reliable
  estimates. Error is reported by price band partly so these gaps are visible
  rather than averaged away.
- **Over-trust in a single number.** The UI pairs every estimate with measured
  band error and an explicit list of what the model cannot see.
- **Data provenance.** The dataset is public and ODbL-licensed, scraped from
  public listings. It contains no personal data — no names, contacts or
  registration numbers.
- **Security.** joblib artefacts are pickles and execute code on load. The
  application loads only the file produced by this project's own training
  script and offers no upload path. This is deliberate and should not be
  changed.

---

## Reproducing

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train_model.py
python -m pytest
```

`random_state=42` throughout; runs are deterministic on the same library
versions. Every figure in this card is read from `reports/evaluation.json`,
which the training script writes.

---

## Maintenance

| | |
|---|---|
| **Status** | Complete; a learning portfolio project, not actively maintained |
| **Last trained** | October 2026, on data with observation year 2020 |
| **Retraining trigger** | Would need a newer dataset — the current one cannot be refreshed |
| **Known issue** | The six-year gap between the data's period and today grows; the model does not and cannot account for it |
| **Contact** | Via the repository's GitHub issues |
