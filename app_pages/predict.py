"""Predict Price page: the input form and the estimate."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ucpp import app_support as sup
from ucpp import config
from ucpp.predict import predict_price


def _default_index(schema: dict, column: str, options: list[str]) -> int:
    """Index of the most common training value, for use as a form default.

    Defaulting to the alphabetically first option would open the form on
    Fuel = CNG (under 1% of listings) and Transmission = Automatic, so a user
    who only changed the brand would get a quote for a rare configuration
    without realising it. Falling back to 0 keeps the form working if an older
    schema has no mode recorded.
    """
    mode = schema.get("category_modes", {}).get(column)
    if mode in options:
        return options.index(mode)
    return 0


def _owner_options(schema: dict) -> list[str]:
    """Order ownership levels sensibly rather than alphabetically."""
    preferred = [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car",
    ]
    available = schema["categories"]["owner"]
    return [o for o in preferred if o in available] + [
        o for o in available if o not in preferred
    ]


def render() -> None:
    st.header("Predict a used-car price")
    st.caption(sup.HISTORICAL_NOTICE)

    pipeline, schema = sup.get_model()
    ranges = schema["numeric_ranges"]
    cats = schema["categories"]

    # Deliberately NOT wrapped in st.form. Widgets inside a form do not trigger
    # a rerun until the form is submitted, which would leave the dependent
    # Model dropdown permanently empty after choosing a brand. Plain widgets
    # rerun on change, so the brand -> model dependency works.
    st.subheader("Vehicle details")

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox(
                config.FEATURE_LABELS["brand"],
                options=sorted(schema["brand_to_models"].keys()),
                index=None,
                placeholder="Choose a brand",
                help="Manufacturer, taken from the listing title.",
            )

            # Dependent dropdown: the model list is filtered by the chosen
            # brand, so the form cannot produce a Maruti/BMW combination that
            # never existed in the data.
            if brand:
                model_options = schema["brand_to_models"][brand]
            else:
                model_options = []
            model_family = st.selectbox(
                config.FEATURE_LABELS["model_family"],
                options=model_options,
                index=None,
                placeholder="Choose a brand first" if not brand else "Choose a model",
                disabled=not brand,
                help="Model family. Trim levels are not modelled separately.",
            )

            year = st.number_input(
                "Manufacturing year",
                min_value=int(config.MIN_PLAUSIBLE_YEAR),
                max_value=int(config.OBSERVATION_YEAR),
                value=2015,
                step=1,
                help=(
                    f"Capped at {config.OBSERVATION_YEAR}, the newest model year "
                    f"in the training data. Age is measured against "
                    f"{config.OBSERVATION_YEAR}, not the current year."
                ),
            )

            km_driven = st.number_input(
                config.FEATURE_LABELS["km_driven"],
                min_value=0,
                max_value=500_000,
                value=50_000,
                step=5_000,
                help=(
                    f"Typical training range: "
                    f"{ranges['km_driven']['p01']:,.0f} to "
                    f"{ranges['km_driven']['p99']:,.0f} km."
                ),
            )

        with col2:
            fuel = st.selectbox(
                config.FEATURE_LABELS["fuel"],
                options=cats["fuel"],
                index=_default_index(schema, "fuel", cats["fuel"]),
            )
            transmission = st.selectbox(
                config.FEATURE_LABELS["transmission"],
                options=cats["transmission"],
                index=_default_index(schema, "transmission", cats["transmission"]),
            )
            owner_opts = _owner_options(schema)
            owner = st.selectbox(
                config.FEATURE_LABELS["owner"],
                options=owner_opts,
                index=_default_index(schema, "owner", owner_opts),
            )
            seller_type = st.selectbox(
                config.FEATURE_LABELS["seller_type"],
                options=cats["seller_type"],
                index=_default_index(schema, "seller_type", cats["seller_type"]),
                help="Who is listing the car. Dealers price differently to owners.",
            )

    with st.expander("Engine and specification (optional)"):
        st.caption(
            "Leave these at their defaults if you do not know them - the "
            "pipeline imputes the training median. Engine power is the "
            "second most important feature, so supplying it helps."
        )
        spec1, spec2, spec3 = st.columns(3)
        with spec1:
            max_power_bhp = st.number_input(
                config.FEATURE_LABELS["max_power_bhp"],
                min_value=0.0,
                max_value=1000.0,
                value=float(ranges["max_power_bhp"]["median"]),
                step=5.0,
            )
        with spec2:
            engine_cc = st.number_input(
                config.FEATURE_LABELS["engine_cc"],
                min_value=0.0,
                max_value=10000.0,
                value=float(ranges["engine_cc"]["median"]),
                step=100.0,
            )
        with spec3:
            mileage_kmpl = st.number_input(
                config.FEATURE_LABELS["mileage_kmpl"],
                min_value=0.0,
                max_value=50.0,
                value=float(ranges["mileage_kmpl"]["median"]),
                step=0.5,
            )
        seats = st.select_slider(
            config.FEATURE_LABELS["seats"],
            options=[2, 4, 5, 6, 7, 8, 9, 10, 14],
            value=5,
        )

    if st.button("Estimate price", type="primary"):
        st.session_state["has_estimated"] = True

    # Remembered in session state so the estimate survives the reruns that
    # changing an input causes - otherwise the result would vanish the moment
    # the user adjusted anything.
    if not st.session_state.get("has_estimated"):
        st.info("Fill in the details above and select **Estimate price**.")
        return

    if not brand or not model_family:
        st.warning("Please choose both a brand and a model to get an estimate.")
        return

    values = {
        "brand": brand,
        "model_family": model_family,
        "vehicle_age": config.OBSERVATION_YEAR - int(year),
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage_kmpl": mileage_kmpl,
        "engine_cc": engine_cc,
        "max_power_bhp": max_power_bhp,
        "seats": seats,
    }

    result = predict_price(pipeline, schema, values)
    report = sup.get_evaluation_report()

    st.divider()
    st.subheader("Estimated price")

    left, right = st.columns([2, 3])
    with left:
        st.metric(
            label=f"{brand} {model_family.split(' ', 1)[-1]} ({int(year)})",
            value=sup.format_inr(result.price_inr),
            help="Point estimate. No confidence interval is shown - see below.",
        )
        st.caption(f"approximately {sup.format_lakh(result.price_inr)}")
        st.caption(f"Vehicle age used: {values['vehicle_age']} years "
                   f"(relative to {config.OBSERVATION_YEAR})")

    with right:
        if report:
            band_rows = report["error_by_price_band"]
            band = _matching_band(result.price_inr, band_rows)
            if band:
                st.markdown(
                    f"**Typical error for cars in this price range** "
                    f"({band['price_band']})"
                )
                st.markdown(
                    f"- Mean absolute error: **{sup.format_inr(band['mae_inr'])}**\n"
                    f"- Median error: **{band['median_abs_pct_error']:.1f}%** "
                    f"of the actual price\n"
                    f"- Measured on {band['n']} held-out listings"
                )
                st.caption(
                    "This is the model's measured error on unseen data in this "
                    "band - not a confidence interval for your specific car, "
                    "which this model cannot provide."
                )

    for warning in result.warnings:
        st.warning(warning, icon=":material/warning:")

    if not result.is_extrapolation:
        st.success(
            "All inputs fall within the ranges and categories the model was "
            "trained on.",
            icon=":material/check_circle:",
        )

    _render_depreciation_curve(pipeline, schema, values, int(year))

    with st.expander("What this estimate does not tell you"):
        st.markdown(
            f"""
- **It is a historical figure.** The training data's newest model year is
  {config.OBSERVATION_YEAR}. Indian used-car prices have moved since; this
  number is not adjusted for that.
- **It is an asking price, not a sale price.** The dataset records what
  sellers listed vehicles at, which is typically above what they accepted.
- **There is no confidence interval.** A single tree-ensemble prediction has no
  calibrated uncertainty, so quoting a range would be inventing one. The
  measured test-set error above is the honest substitute.
- **Condition is not an input.** Accident history, service record, tyres,
  paintwork and location are absent from the data and can easily move a real
  price by more than the model's average error.
- **Trim is not modelled.** The model knows "{model_family}", not which
  variant, and trim differences within a model family can be substantial.
"""
        )


def _matching_band(price: float, band_rows: list[dict]) -> dict | None:
    """Find the error-breakdown row whose band contains this price."""
    for (label, low, high), row in zip(config.PRICE_BANDS, band_rows):
        if low <= price < high and row["price_band"] == label:
            return row
    # Band lists can differ in length if a band was empty in the test set.
    for row in band_rows:
        for label, low, high in config.PRICE_BANDS:
            if row["price_band"] == label and low <= price < high:
                return row
    return None


def _render_depreciation_curve(pipeline, schema, values: dict, chosen_year: int) -> None:
    """Show how the estimate varies with model year, holding all else fixed.

    More useful than a bare number: it shows the shape the model has learned
    for this particular car, and makes any odd kink visible instead of hidden.
    """
    years = list(range(2000, config.OBSERVATION_YEAR + 1))
    rows = []
    for year in years:
        trial = dict(values)
        trial["vehicle_age"] = config.OBSERVATION_YEAR - year
        rows.append(
            {
                "year": year,
                "price": predict_price(pipeline, schema, trial).price_inr,
            }
        )
    curve = pd.DataFrame(rows)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=curve["year"],
            y=curve["price"],
            mode="lines",
            line=dict(color=sup.ACCENT, width=3),
            name="Predicted price",
            hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>",
        )
    )
    selected = curve.loc[curve["year"] == chosen_year]
    if not selected.empty:
        figure.add_trace(
            go.Scatter(
                x=selected["year"],
                y=selected["price"],
                mode="markers",
                marker=dict(color=sup.WARN, size=13, line=dict(color="white", width=2)),
                name="Your selection",
                hovertemplate="Selected %{x}: ₹%{y:,.0f}<extra></extra>",
            )
        )

    figure.update_layout(
        template=sup.PLOT_TEMPLATE,
        title="Predicted price by model year (all other inputs unchanged)",
        xaxis_title="Manufacturing year",
        yaxis_title="Predicted price (INR)",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    figure.update_yaxes(tickprefix="₹", tickformat=",.0f")

    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "A model-year sweep for this exact configuration. Flat segments mean "
        "the training data had few examples of that model at those ages, so "
        "the trees fall back on neighbouring years."
    )
