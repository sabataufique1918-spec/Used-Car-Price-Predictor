"""Train, compare and evaluate the price models, then save the best pipeline.

Protocol, in the order it runs:

  1. Split once into train/test with GroupShuffleSplit, grouped on vehicle
     identity, so a relisted car cannot appear on both sides.
  2. Select between candidate models using grouped cross-validation on the
     TRAINING data only.
  3. Tune the winner's hyper-parameters, again on training folds only.
  4. Touch the test set exactly once, at the end, to report final numbers.

Step 4 is the whole point of steps 1-3. Any peek at the test set during
selection turns the final figure into a training score wearing a disguise.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --quick     # skip hyper-parameter search
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    cross_val_predict,
)

from ucpp import config, data_prep, evaluate, features, model_io  # noqa: E402
from ucpp.console import enable_utf8_stdout  # noqa: E402

enable_utf8_stdout()

N_CV_FOLDS = 5


def split_grouped(clean_df: pd.DataFrame):
    """Split off a held-out test set, keeping each vehicle wholly on one side.

    A plain ``train_test_split`` would scatter the ~350 relisted vehicles
    across both sets. The model would then be scored partly on cars whose twin
    it had memorised, which inflates every metric while teaching us nothing
    about a genuinely new listing.
    """
    groups = data_prep.vehicle_identity(clean_df)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(clean_df, groups=groups))

    train = clean_df.iloc[train_idx].reset_index(drop=True)
    test = clean_df.iloc[test_idx].reset_index(drop=True)

    # Assert the guarantee rather than trusting it.
    train_groups = set(data_prep.vehicle_identity(train))
    test_groups = set(data_prep.vehicle_identity(test))
    overlap = train_groups & test_groups
    assert not overlap, f"Vehicle identity leaked across the split: {len(overlap)} groups"

    return train, test


def select_parsimonious(search: GridSearchCV) -> tuple[dict, int]:
    """Pick the simplest model whose CV score ties the best one statistically.

    This is the classic "one standard error rule". Grid search returns whatever
    scored highest, but across 5 folds the gap between the top few candidates
    is usually smaller than the fold-to-fold noise. Taking the raw winner then
    buys an unmeasurable accuracy gain at a real cost: 600 trees at
    min_samples_leaf=2 produced a 61 MB artefact, which is slow to load on a
    free host and awkward in git, for ~0.06% lower CV MAE than 300 trees.

    So: compute the standard error of the best candidate across folds, keep
    every candidate within one SE of it, and from those take the one with the
    fewest trees and the largest leaves. The accuracy given up sits inside the
    noise band we can actually measure; the artefact gets several times smaller.

    Returns the chosen parameters (keys stripped of the pipeline prefix) and
    the index of that candidate in ``cv_results_``.
    """
    results = search.cv_results_
    means = results["mean_test_score"]  # negative MAE, so higher is better
    stds = results["std_test_score"]
    best_idx = int(np.argmax(means))

    standard_error = stds[best_idx] / np.sqrt(search.n_splits_)
    threshold = means[best_idx] - standard_error
    tied = [i for i in range(len(means)) if means[i] >= threshold]

    def complexity(i: int) -> tuple:
        params = results["params"][i]
        # Fewer trees first, then larger leaves (shallower, smaller trees).
        return (
            params.get("model__regressor__n_estimators", 0),
            -params.get("model__regressor__min_samples_leaf", 0),
        )

    chosen_idx = min(tied, key=complexity)

    print(f"\n1-SE rule: best CV MAE {data_prep.format_inr(-means[best_idx])}, "
          f"standard error {data_prep.format_inr(standard_error)}")
    print(f"  {len(tied)} of {len(means)} candidates are within one SE of best.")

    chosen = {
        k.replace("model__regressor__", ""): v
        for k, v in results["params"][chosen_idx].items()
    }
    return chosen, chosen_idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="skip the hyper-parameter search"
    )
    args = parser.parse_args()

    print("=" * 72)
    print("TRAINING")
    print("=" * 72)

    clean_df = data_prep.load_clean()
    print(f"\nClean dataset: {len(clean_df):,} rows")

    # ---------------------------------------------------------------- 1. split
    train, test = split_grouped(clean_df)
    X_train, y_train = train[config.FEATURES], train[config.TARGET]
    X_test, y_test = test[config.FEATURES], test[config.TARGET]
    groups_train = data_prep.vehicle_identity(train)

    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows "
          f"({100 * len(test) / len(clean_df):.0f}% held out)")
    print("Split is grouped on vehicle identity; no vehicle spans both sets.")
    print("The test set is NOT touched again until the final section.")

    # ------------------------------------------------- 2. model selection (CV)
    print("\n" + "-" * 72)
    print(f"MODEL COMPARISON - grouped {N_CV_FOLDS}-fold CV on TRAINING data only")
    print("-" * 72)

    cv = GroupKFold(n_splits=N_CV_FOLDS)
    comparison = {}

    for name, builder in features.MODEL_BUILDERS.items():
        pipeline = builder()
        start = time.time()
        # cross_val_predict refits the pipeline on each fold, so imputers and
        # encoders are learned from that fold's training part only.
        preds = cross_val_predict(
            pipeline, X_train, y_train, cv=cv, groups=groups_train, n_jobs=1
        )
        elapsed = time.time() - start
        metrics = evaluate.regression_metrics(y_train, preds)
        metrics["fit_seconds"] = round(elapsed, 1)
        comparison[name] = metrics

        print(
            f"\n{name:<18} {features.MODEL_DESCRIPTIONS[name]}"
            f"\n  MAE  {data_prep.format_inr(metrics['mae_inr']):>14}"
            f"   RMSE {data_prep.format_inr(metrics['rmse_inr']):>14}"
            f"\n  R2   {metrics['r2']:>14.4f}"
            f"   median abs % err {metrics['median_abs_pct_error']:>6.1f}%"
            f"   ({elapsed:.1f}s)"
        )

    baseline_mae = comparison["baseline_median"]["mae_inr"]
    print(f"\nBaseline to beat (median price): MAE "
          f"{data_prep.format_inr(baseline_mae)}")
    for name, m in comparison.items():
        if name == "baseline_median":
            continue
        lift = evaluate.improvement_over_baseline(m["mae_inr"], baseline_mae)
        print(f"  {name:<18} {lift:>6.1f}% lower MAE than baseline")

    candidates = {k: v for k, v in comparison.items() if k != "baseline_median"}
    best_name = min(candidates, key=lambda k: candidates[k]["mae_inr"])
    print(f"\nSelected by CV MAE: {best_name}")

    # ------------------------------------------------------ 3. tune the winner
    best_params = {}
    if args.quick or best_name == "ridge":
        print("\nSkipping hyper-parameter search "
              f"({'--quick' if args.quick else 'linear model, defaults are fine'}).")
        final_pipeline = features.MODEL_BUILDERS[best_name]()
    else:
        print("\n" + "-" * 72)
        print("HYPER-PARAMETER SEARCH - grouped CV on TRAINING data only")
        print("-" * 72)
        # Small, CPU-friendly grid. min_samples_leaf controls overfitting;
        # max_features controls how decorrelated the trees are.
        grid = {
            "model__regressor__n_estimators": [300, 600],
            "model__regressor__min_samples_leaf": [1, 2, 4],
            "model__regressor__max_features": [1.0, 0.5],
        }
        search = GridSearchCV(
            features.MODEL_BUILDERS[best_name](),
            grid,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
            verbose=1,
        )
        search.fit(X_train, y_train, groups=groups_train)
        lowest = {
            k.replace("model__regressor__", ""): v
            for k, v in search.best_params_.items()
        }
        print(f"\nLowest-MAE params: {lowest}")
        print(f"  CV MAE {data_prep.format_inr(-search.best_score_)}")

        best_params, chosen_idx = select_parsimonious(search)
        chosen_mae = -search.cv_results_["mean_test_score"][chosen_idx]
        print(f"Chosen params (1-SE rule): {best_params}")
        print(f"  CV MAE {data_prep.format_inr(chosen_mae)}")
        final_pipeline = features.MODEL_BUILDERS[best_name](**best_params)

    # ------------------------------------------- 4. fit on train, score ONCE
    print("\n" + "-" * 72)
    print("FINAL EVALUATION - held-out test set, touched for the first time")
    print("-" * 72)

    final_pipeline.fit(X_train, y_train)
    test_pred = final_pipeline.predict(X_test)
    test_metrics = evaluate.regression_metrics(y_test, test_pred)

    baseline = features.build_baseline().fit(X_train, y_train)
    baseline_test = evaluate.regression_metrics(y_test, baseline.predict(X_test))

    print(f"\n{'':<26}{'baseline':>16}{best_name:>18}")
    print(f"{'MAE (INR)':<26}{data_prep.format_inr(baseline_test['mae_inr']):>16}"
          f"{data_prep.format_inr(test_metrics['mae_inr']):>18}")
    print(f"{'RMSE (INR)':<26}{data_prep.format_inr(baseline_test['rmse_inr']):>16}"
          f"{data_prep.format_inr(test_metrics['rmse_inr']):>18}")
    print(f"{'R2':<26}{baseline_test['r2']:>16.4f}{test_metrics['r2']:>18.4f}")
    print(f"{'Median abs % error':<26}"
          f"{baseline_test['median_abs_pct_error']:>15.1f}%"
          f"{test_metrics['median_abs_pct_error']:>17.1f}%")

    lift = evaluate.improvement_over_baseline(
        test_metrics["mae_inr"], baseline_test["mae_inr"]
    )
    print(f"\nMAE is {lift:.1f}% lower than the median baseline on held-out data.")

    print("\n--- Error by actual price band ---")
    band_df = evaluate.error_by_price_band(y_test, test_pred)
    print(band_df.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    print("\nNote: rupee error grows with price, as expected. The percentage")
    print("column is the fairer read of accuracy across bands.")

    print("\n--- Permutation importance (test set, MAE increase when shuffled) ---")
    importance = evaluate.permutation_importance_frame(
        final_pipeline, X_test, y_test, n_repeats=5
    )
    print(importance.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    # ----------------------------------------------------------------- save
    metrics_payload = {
        "observation_year": config.OBSERVATION_YEAR,
        "dataset": {
            "source": config.KAGGLE_DATASET_REF,
            "file": config.RAW_CSV_NAME,
            "licence": config.DATASET_LICENCE,
            "clean_rows": int(len(clean_df)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "split": "GroupShuffleSplit on name+year+km_driven",
        },
        "selected_model": best_name,
        "best_params": best_params,
        "cv_comparison": comparison,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_test,
        "mae_improvement_over_baseline_pct": lift,
        "error_by_price_band": band_df.to_dict(orient="records"),
        "permutation_importance": importance.to_dict(orient="records"),
    }

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    # Schema is built from the FULL clean dataset so the app's dropdowns offer
    # every real brand/model, not only those that landed in the training split.
    # This is presentation metadata, not fitted state - no leakage.
    schema = model_io.build_schema(clean_df, best_name, metrics=test_metrics)
    model_path, schema_path = model_io.save_model(final_pipeline, schema)

    size_mb = model_path.stat().st_size / 1e6
    print(f"\nWrote {config.METRICS_PATH}")
    print(f"Wrote {model_path} ({size_mb:.1f} MB)")
    if size_mb > 50:
        print("  WARNING: artefacts over 50 MB trigger a GitHub size warning "
              "and slow cold starts on Streamlit Community Cloud. Raise "
              "min_samples_leaf or lower n_estimators.")
    print(f"Wrote {schema_path}")
    print("\nNext:  streamlit run app.py")


if __name__ == "__main__":
    main()
