"""Model Evaluation page: the measured results, read from the saved report.

Nothing here is recomputed in the app. Every figure comes from
``reports/evaluation.json``, written by the training script against the
held-out test set.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ucpp import app_support as sup
from ucpp import config


def render() -> None:
    st.header("Model Evaluation")

    report = sup.get_evaluation_report()
    if report is None:
        st.warning(
            "No evaluation report found. Run `python scripts/train_model.py` "
            "to generate `reports/evaluation.json`."
        )
        return

    test = report["test_metrics"]
    baseline = report["baseline_test_metrics"]
    dataset = report["dataset"]

    st.caption(
        f"All figures below are measured on **{dataset['test_rows']:,} held-out "
        f"listings** that no model saw during training, selection or tuning. "
        f"Selected model: **{report['selected_model']}**."
    )

    # ------------------------------------------------------------- headline
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Mean absolute error",
        sup.format_inr(test["mae_inr"]),
        delta=f"{-report['mae_improvement_over_baseline_pct']:.1f}% vs baseline",
        delta_color="inverse",
        help="Average size of the miss, in rupees. Lower is better.",
    )
    c2.metric(
        "RMSE",
        sup.format_inr(test["rmse_inr"]),
        help=(
            "Root mean squared error. Penalises large misses much more than "
            "MAE, so the gap between the two shows how heavy the tail of bad "
            "predictions is."
        ),
    )
    c3.metric(
        "R²",
        f"{test['r2']:.4f}",
        help="Share of price variance explained on held-out data.",
    )
    c4.metric(
        "Median error",
        f"{test['median_abs_pct_error']:.1f}%",
        help="Half of predictions are within this percentage of the real price.",
    )

    st.info(
        f"**Read this before quoting the R².** The median baseline — predicting "
        f"{sup.format_inr(0) if False else 'the training median'} for every car "
        f"— gets MAE {sup.format_inr(baseline['mae_inr'])} and R² "
        f"{baseline['r2']:.3f}. The model reduces MAE by "
        f"**{report['mae_improvement_over_baseline_pct']:.1f}%**. That "
        f"comparison, not the R² alone, is the real result.",
        icon=":material/info:",
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Model comparison", "Error by price band", "Feature importance", "Raw report"]
    )

    # -------------------------------------------------------- model comparison
    with tab1:
        st.subheader("Candidate models, grouped 5-fold CV on training data")
        st.caption(
            "Models were compared using cross-validation **within the training "
            "set only**. The held-out test set was scored once, at the end, "
            "for the winner alone."
        )

        rows = []
        for name, metrics in report["cv_comparison"].items():
            rows.append(
                {
                    "Model": name,
                    "CV MAE (INR)": metrics["mae_inr"],
                    "CV RMSE (INR)": metrics["rmse_inr"],
                    "CV R²": metrics["r2"],
                    "Median abs % error": metrics["median_abs_pct_error"],
                    "Fit time (s)": metrics.get("fit_seconds", float("nan")),
                }
            )
        comparison = pd.DataFrame(rows).sort_values("CV MAE (INR)")

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
            column_config={
                "CV MAE (INR)": st.column_config.NumberColumn(format="%.0f"),
                "CV RMSE (INR)": st.column_config.NumberColumn(format="%.0f"),
                "CV R²": st.column_config.NumberColumn(format="%.4f"),
                "Median abs % error": st.column_config.NumberColumn(format="%.1f%%"),
                "Fit time (s)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        plot_df = comparison.copy()
        figure = px.bar(
            plot_df,
            x="Model",
            y="CV MAE (INR)",
            color="Model",
            color_discrete_sequence=[sup.ACCENT, sup.ACCENT_SOFT, sup.NEUTRAL, sup.WARN],
        )
        figure.update_layout(
            template=sup.PLOT_TEMPLATE,
            height=380,
            showlegend=False,
            title="Cross-validated MAE by model (lower is better)",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        figure.update_yaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(figure, use_container_width=True)

        if report.get("best_params"):
            st.markdown(
                f"**Chosen hyper-parameters:** `{report['best_params']}`  \n"
                "Selected with the one-standard-error rule: the simplest "
                "settings statistically tied with the best-scoring ones, which "
                "keeps the saved model small without a measurable accuracy cost."
            )

    # ------------------------------------------------------ error by price band
    with tab2:
        st.subheader("Where the model works, and where it does not")
        bands = pd.DataFrame(report["error_by_price_band"])

        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=bands["price_band"],
                y=bands["mae_inr"],
                name="MAE (INR)",
                marker_color=sup.ACCENT,
                hovertemplate="%{x}<br>MAE ₹%{y:,.0f}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=bands["price_band"],
                y=bands["median_abs_pct_error"],
                name="Median abs % error",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color=sup.WARN, width=3),
                marker=dict(size=10),
                hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
            )
        )
        figure.update_layout(
            template=sup.PLOT_TEMPLATE,
            height=430,
            title="Absolute error grows with price; percentage error does not",
            yaxis=dict(title="MAE (INR)", tickprefix="₹", tickformat=",.0f"),
            yaxis2=dict(
                title="Median abs % error", overlaying="y", side="right",
                ticksuffix="%", showgrid=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=10, r=10, t=70, b=10),
        )
        st.plotly_chart(figure, use_container_width=True)

        st.dataframe(
            bands.rename(
                columns={
                    "price_band": "Price band",
                    "n": "Listings",
                    "share_of_test_pct": "Share of test",
                    "mae_inr": "MAE (INR)",
                    "rmse_inr": "RMSE (INR)",
                    "median_abs_pct_error": "Median abs % error",
                    "mean_bias_inr": "Mean bias (INR)",
                }
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "MAE (INR)": st.column_config.NumberColumn(format="%.0f"),
                "RMSE (INR)": st.column_config.NumberColumn(format="%.0f"),
                "Median abs % error": st.column_config.NumberColumn(format="%.1f%%"),
                "Mean bias (INR)": st.column_config.NumberColumn(format="%.0f"),
                "Share of test": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

        st.markdown(
            """
**What the bias column shows.** A positive mean bias means the model
over-predicts that band on average; negative means it under-predicts. The
pattern — over-predicting the cheapest cars and under-predicting the most
expensive — is regression to the mean, and it is the single most important
limitation to know about when reading an individual estimate. Cars at the
extremes of the price range are where the model is least trustworthy.
"""
        )

    # ------------------------------------------------------ feature importance
    with tab3:
        st.subheader("Permutation importance on the test set")
        importance = pd.DataFrame(report["permutation_importance"])
        importance["label"] = importance["feature"].map(
            lambda f: config.FEATURE_LABELS.get(f, f)
        )
        importance = importance.sort_values("mae_increase_inr")

        figure = px.bar(
            importance,
            x="mae_increase_inr",
            y="label",
            orientation="h",
            error_x="std",
            color_discrete_sequence=[sup.ACCENT],
            labels={
                "mae_increase_inr": "Increase in MAE when this feature is shuffled",
                "label": "",
            },
        )
        figure.update_layout(
            template=sup.PLOT_TEMPLATE,
            height=460,
            title="How much each input actually matters",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        figure.update_xaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(figure, use_container_width=True)

        st.markdown(
            """
Each bar is how much worse the model's MAE gets when that one column is
randomly shuffled, which breaks its relationship with price while leaving
everything else intact.

Permutation importance is reported rather than the tree ensemble's built-in
`feature_importances_`, because impurity-based importance is systematically
biased towards high-cardinality features — which would flatter `model_family`
and its 210 categories.
"""
        )

    with tab4:
        st.caption(
            "The full saved report, exactly as written by the training script."
        )
        st.json(report, expanded=False)


