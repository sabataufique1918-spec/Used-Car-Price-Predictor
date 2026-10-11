# Portfolio kit

Supporting material for presenting this project: a demo script, interview
preparation, and a LinkedIn draft. **Every number here is taken from
`reports/evaluation.json`** — nothing is estimated or rounded up.

---

## 1. Sixty-second demo script

> Timings assume you drive the app while talking. Practise once; the model-year
> chart takes a moment to render.

**[0:00–0:10] The problem and the result**

> "Used-car pricing in India is guesswork for most private sellers. I trained a
> model on about 6,900 real CarDekho listings that estimates a car's asking
> price. On held-out data it's off by about ₹78,000 on average — that's 74%
> better than just guessing the median price, with an R² of 0.93."

**[0:10–0:25] Live prediction** — *Predict Price page*

> "Pick a brand — the model dropdown filters to just that brand's models. Say a
> 2015 Maruti Swift, 50,000 kilometres, petrol, manual, first owner."

*Click Estimate price.*

> "₹4.75 lakh. And next to it, the model's actual measured error for cars in
> that price band — about 10%. That's measured on held-out data, not a
> confidence interval I invented."

**[0:25–0:35] The honesty bit** — *scroll to the chart and the caveats*

> "This curve shows how the estimate moves with model year for this exact car.
> And I'm explicit about what the model can't see: condition, accident history,
> trim, location. It's also historical — the data stops at 2020, so this is a
> 2020 asking price, not today's market."

**[0:35–0:50] The engineering** — *Model Evaluation page*

> "Here's what I think matters most. I compared four models against a median
> baseline — because an R² of 0.93 means nothing if a trivial model gets 0.90.
>
> And this breakdown by price band is the real story: the model is solid in the
> ₹2–7 lakh range where the data is dense, but it over-prices cheap cars and
> under-prices expensive ones by about ₹1.9 lakh. That's regression to the
> mean, and I'd rather show it than hide it behind one average number."

**[0:50–1:00] The catch worth mentioning**

> "One thing I'm glad I caught: the raw file had 1,200 duplicate listings and
> hundreds of relisted cars. A normal random split would have put the same car
> in training and test and inflated every metric. So the split is grouped by
> vehicle identity. That one change is the difference between an honest number
> and a flattering one."

### If they only give you 20 seconds

> "I trained a model on 6,900 Indian used-car listings — ₹78,000 average error,
> R² 0.93, 74% better than a median baseline. The interesting part isn't the
> model, it's that the raw data had 1,200 duplicates that would have leaked
> across the train/test split. I group the split by vehicle identity, and I
> report error by price band because the model's much weaker at the extremes."

---

## 2. Five interview questions, with answers

### Q1. Your R² is 0.93. How do I know that's actually good?

**Answer.**

On its own you don't, which is why I always report it against a baseline.

A median-price baseline — predict ₹4 lakh for every car — gets MAE ₹2,98,392
and R² of −0.061 on the same test set. My model gets MAE ₹78,092 and R² 0.928.
That's a **73.8% reduction in mean absolute error**, and that comparison is the
real result.

Two details I'd add:

- The baseline's R² is *negative*, which surprises people. It's not a bug: the
  median minimises absolute error, not squared error, so it scores slightly
  worse than predicting the mean. R² is zero for the mean, not the median.
- R² is also flattered by the price spread here. Prices run from ₹30,000 to
  ₹1 crore, so a model that merely separates hatchbacks from luxury SUVs
  already captures a lot of variance. That's why I lead with MAE in rupees and
  with median percentage error — 12.0% — which are harder to inflate.

---

### Q2. Walk me through how you prevented data leakage.

**Answer.**

Three separate places, because leakage gets in through more than one door.

**1. Duplicate listings across the split.** This was the big one. The raw file
has **1,202 byte-identical duplicate rows** — about 15% of the data, from the
scrape republishing listings. On top of that, after removing those, **348
vehicles still appear more than once** (747 rows): same name, same year, same
odometer, slightly different price. Almost certainly one car relisted.

A plain `train_test_split` scatters those across both sides. The model then
gets scored partly on cars whose near-twin it memorised, and every metric comes
out inflated — you'd never know from the numbers.

So I drop exact duplicates before splitting, then use `GroupShuffleSplit` keyed
on `name + year + km_driven`. The same grouping drives every CV fold, and the
training script asserts zero group overlap before it continues.

**2. Preprocessing fitted on test data.** Imputation medians and one-hot
category lists are both learned from data. If you fit them on the full dataset
before splitting, test information bleeds into training. I put everything
inside a scikit-learn `Pipeline`, so the only way to fit the imputer is to call
`fit`, which only ever sees the training fold. It's structural rather than
disciplinary — I can't get it wrong by forgetting.

**3. Tuning against the test set.** I selected the model and tuned
hyper-parameters using grouped 5-fold CV *within the training set*, and touched
the test set exactly once at the very end. If you grid-search against your test
set, it stops being a test set and becomes a very expensive validation set.

I'd also flag a fourth that I avoided by feature choice: `car data.csv` in the
same Kaggle download has a `Present_Price` column — the current showroom price.
Using it would give a spectacular R² and be completely useless, because you
don't know it at prediction time.

---

### Q3. Why Extra Trees? And why does the odometer reading barely matter?

**Answer.**

**On model choice** — I compared four, by cross-validated MAE:

| Model | CV MAE |
|---|---|
| Median baseline | ₹2,72,796 |
| Ridge | ₹80,132 |
| Random Forest | ₹76,869 |
| **Extra Trees** | **₹75,518** |

Extra Trees won, but honestly the margin over Ridge is about 6% — most of the
price signal is close to log-linear in age and engine power. I'd have been
comfortable shipping Ridge; it trains in 0.3 seconds and the artefact is a few
kilobytes. Extra Trees handles the interactions a bit better, and its extra
split randomness lowers variance relative to Random Forest.

One real bug worth mentioning: my first run had Ridge *beating* both ensembles
by a wide margin, which should never happen on data like this. The cause was
`max_features="sqrt"` — that's the *classification* default. On a ~110-column
one-hot matrix it gave each split only about 10 candidate columns, mostly
sparse model dummies, so the trees rarely got to split on age or power at all.
Setting it to `1.0`, the regression default, moved Random Forest's R² from 0.76
to 0.89. A suspicious result is worth chasing.

**On the odometer** — this is the finding I like most. Permutation importance
on the test set:

| Feature | MAE increase when shuffled |
|---|---|
| Vehicle age | ₹1,55,403 |
| Max power | ₹94,186 |
| Engine cc | ₹43,581 |
| **Kilometres driven** | **₹7,258** |

Shuffling the odometer costs almost nothing. The reason is that age and
kilometres are strongly correlated — Indian cars average roughly 10–15,000 km a
year — so once the model knows the age, the odometer carries very little
*additional* information. It's not that mileage doesn't affect price; it's that
it's largely redundant given age.

That has a practical consequence: this model can't really distinguish a
garage-kept car from a taxi of the same age, which is part of why I'm cautious
about the estimates.

I used permutation importance rather than the built-in
`feature_importances_` deliberately, since impurity importance is biased toward
high-cardinality features and would have overstated `model_family` and its 210
categories.

---

### Q4. Why doesn't the app show a confidence interval?

**Answer.**

Because I don't have one, and showing a fake one would be worse than showing
nothing.

A single point prediction from a tree ensemble has no calibrated uncertainty
attached. There are things people do — take the spread of predictions across
the individual trees, or use quantile regression forests — but tree spread
measures *model variance*, not predictive uncertainty, and it's known to be
badly under-calibrated. Printing "±10%" because it looks professional would be
inventing a number, and someone might make a financial decision on it.

What I show instead is the model's **measured error on held-out data for that
price band**. If the estimate is ₹4.75 lakh, the app says: for cars in the ₹4–7
lakh band, MAE was ₹66,448 and the median error was 10.0%, measured on 433
test listings. That's an empirical, honest statement.

I felt strongly enough about it to write a test that fails if anyone ever adds
a `confidence`, `ci_low` or `margin` field to the prediction result. It's a
guard against a future well-meaning edit.

If I genuinely needed intervals, I'd train a quantile regression forest or use
conformal prediction, which gives distribution-free coverage guarantees — and
I'd validate the coverage before showing anything.

---

### Q5. This model is trained on data ending in 2020. Why is that fine, and what would you do about it?

**Answer.**

It isn't fine for real use, and I've been explicit about that rather than
quietly shipping it.

The first thing it changed is a correctness issue, not just a caveat. Vehicle
age has to be computed against the **dataset's observation year, 2020** — not
`date.today().year`. If I'd used the current year, every car would carry six
years of phantom age, and every input would fall outside the range the model
was fitted on. The nasty part is that it wouldn't error — tree models just
return whatever they learned at the oldest age they saw, so you'd get
confident, wrong answers. It's pinned in config, the prep script warns if the
data disagrees, and there's a test asserting age doesn't track today's year.

Second, I framed the output honestly throughout. The app says on every page
that these are *historical asking prices*, not current valuations. They're also
asking prices, not transaction prices — sellers list above what they accept —
so they skew high twice over.

**What I'd do with more scope:**

- Get data with a `listing_date` column and model price as a function of both
  vehicle age and calendar time, so the trend is separated from depreciation.
- Validate against a recent sample before claiming any present-day accuracy.
- Add drift monitoring. Right now the model has no idea it's stale — and it
  already is.
- Collect location, since the chosen file has no city column and regional
  variation in India is substantial.

The honest summary is that this is a well-engineered model of a market that no
longer exists. As a demonstration of the workflow that's fine; as a pricing
tool it would need current data.

### Three more they might ask

**"Why log-transform the target?"** Prices span ₹30,000 to ₹1 crore and are
heavily right-skewed. On raw rupees, squared error puts nearly all the model's
effort into a handful of luxury cars. On the log scale it optimises *relative*
error, which is what matters when quoting a price, and predictions can't come
back negative. I still report metrics in rupees, since that's the unit people
care about.

**"How do you handle a brand the model has never seen?"** The encoder uses
`handle_unknown="infrequent_if_exist"` with `min_frequency=15`, so unknown
categories route into the same bucket as rare ones instead of raising. The app
then tells the user the estimate is less reliable. There's a test that feeds an
unseen value into *every* categorical column.

**"Why is the saved model 30 MB and not 61?"** Grid search's top candidate used
600 trees and beat 300 trees by ₹42 of CV MAE — against a fold-to-fold standard
error of ₹1,449. That's noise. I applied the one-standard-error rule: keep
everything within one SE of the best, take the simplest. Cost ₹117 of test MAE,
about 0.15%; halved the artefact.

---

## 3. LinkedIn post draft

> Uses only verified results. Replace the two bracketed placeholders before
> posting.

---

I built a used-car price predictor, and the most useful thing I learned had
nothing to do with the model.

I trained it on ~6,900 real CarDekho listings to estimate asking prices in the
Indian used-car market. The results, on a held-out test set:

• Mean absolute error: ₹78,092
• R²: 0.93
• Median error: 12% of the actual price
• 74% lower error than a median-price baseline

But here's what actually took the time.

**The raw data had 1,202 duplicate listings.** About 15% of the file. On top of
that, 348 vehicles appeared more than once — same car, same year, same odometer,
relisted at a slightly different price.

If I'd used a normal random train/test split, the same car would have landed in
both. My model would have been graded on vehicles it had effectively memorised,
and every metric would have looked better than it deserved. I'd never have
known from the numbers — they'd just have been quietly wrong.

So the split is grouped by vehicle identity. No car can appear on both sides.

Three other things I'd do the same way again:

🔹 **Always build the dumb baseline first.** "R² = 0.93" sounds impressive until
you check what predicting the median gets you. Here the baseline's R² was
*negative* — so the 0.93 was real. But I only know that because I measured it.

🔹 **Report error by segment, not just on average.** My model is accurate to
~10% in the ₹2–7 lakh range where the data is dense. Above ₹12 lakh it
under-prices by ₹1.9 lakh on average. One headline number would have hidden
that completely, and that gap is exactly what a user needs to know.

🔹 **Don't invent a confidence interval.** A tree ensemble's point prediction has
no calibrated uncertainty. Printing "±10%" because it looks professional would
be making up a number someone might act on. The app shows measured test-set
error for the relevant price band instead.

One more, because it's the kind of bug that doesn't announce itself: the newest
car in the dataset is from 2020. Vehicle age had to be computed against **2020**,
not today's year. Get that wrong and every car silently gains six years of age,
and the model returns confident nonsense rather than an error.

Stack: Python, pandas, scikit-learn, Streamlit, Plotly. Preprocessing lives
inside the sklearn Pipeline so it can only ever be fitted on training folds.
80 tests, including Streamlit smoke tests — which caught a form bug that would
have left a dropdown permanently empty for every user.

Code and full model card: [GITHUB LINK]
Live app: [STREAMLIT LINK]

Happy to hear how others handle duplicate detection in scraped datasets — I
suspect I'm still missing cases.

#MachineLearning #DataScience #Python #scikitlearn #Streamlit #MLEngineering

---

### A shorter variant

I built a used-car price predictor on ~6,900 Indian listings.

MAE ₹78,092 · R² 0.93 · 74% better than a median baseline.

The part that mattered wasn't the model. The raw data had 1,202 duplicate
listings and 348 relisted vehicles. A normal random train/test split would have
put the same car on both sides and inflated every metric — invisibly.

Grouping the split by vehicle identity fixed it. That one change is the
difference between a number I can defend and a number that just looks good.

Also: I report error by price band, not just the average (the model is ~10% off
in the ₹2–7 lakh range, but under-prices ₹12 lakh+ cars by ₹1.9 lakh), and I
deliberately show no confidence interval — a tree ensemble doesn't have a
calibrated one, and inventing it would be worse than omitting it.

Python · pandas · scikit-learn · Streamlit · 80 tests

Code: [GITHUB LINK]

#MachineLearning #DataScience #Python
