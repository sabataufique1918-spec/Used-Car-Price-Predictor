"""Data Explorer page: what the model was actually trained on."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from ucpp import app_support as sup
from ucpp import config


def render() -> None:
    st.header("Data Explorer")
    st.caption(
        "The cleaned training data. Worth a look before trusting any "
        "prediction: the model is only confident where this data is dense."
    )

    data = sup.get_clean_data()

    # ---------------------------------------------------------------- filters
    with st.sidebar:
        st.subheader("Filters")
        brands = sorted(data["brand"].dropna().unique())
        selected_brands = st.multiselect(
            "Brand", options=brands, default=[], placeholder="All brands"
        )
        fuels = st.multiselect(
            "Fuel", options=sorted(data["fuel"].dropna().unique()), default=[],
            placeholder="All fuel types",
        )
        year_min, year_max = int(data["year"].min()), int(data["year"].max())
        year_range = st.slider(
            "Model year", year_min, year_max, (year_min, year_max)
        )
        price_cap = st.slider(
            "Max price shown (lakh)",
            min_value=1,
            max_value=int(data[config.TARGET].max() / 100_000) + 1,
            value=25,
            help=(
                "Prices are heavily right-skewed. Capping the display makes the "
                "bulk of the market readable; it does not change the model."
            ),
        )

    view = data.copy()
    if selected_brands:
        view = view[view["brand"].isin(selected_brands)]
    if fuels:
        view = view[view["fuel"].isin(fuels)]
    view = view[view["year"].between(*year_range)]
    view = view[view[config.TARGET] <= price_cap * 100_000]

    if view.empty:
        st.warning("No listings match these filters. Widen them to continue.")
        return

    # ----------------------------------------------------------------- top row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Listings shown", f"{len(view):,}", delta=f"of {len(data):,} total")
    c2.metric("Median price", sup.format_inr(view[config.TARGET].median()))
    c3.metric("Median age", f"{view['vehicle_age'].median():.0f} yrs")
    c4.metric("Median odometer", f"{view['km_driven'].median():,.0f} km")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Price distribution", "Price vs age", "By category", "Raw rows"]
    )

    with tab1:
        figure = px.histogram(
            view, x=config.TARGET, nbins=60,
            color_discrete_sequence=[sup.ACCENT],
            labels={config.TARGET: "Selling price (INR)"},
        )
        figure.update_layout(
            template=sup.PLOT_TEMPLATE, height=400,
            title="Selling price distribution",
            yaxis_title="Number of listings",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        figure.update_xaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            "The long right tail is why the models are trained on log(price): "
            "fitted on raw rupees, squared error would chase a handful of "
            "luxury cars and neglect the dense ₹2-7 lakh market."
        )

    with tab2:
        figure = px.scatter(
            view,
            x="vehicle_age",
            y=config.TARGET,
            color="fuel",
            hover_data=["name", "km_driven", "transmission"],
            opacity=0.55,
            labels={
                "vehicle_age": f"Age in years (relative to {config.OBSERVATION_YEAR})",
                config.TARGET: "Selling price (INR)",
            },
        )
        figure.update_layout(
            template=sup.PLOT_TEMPLATE, height=460,
            title="Price against vehicle age",
            margin=dict(l=10, r=10, t=50, b=10),
            legend_title_text="Fuel",
        )
        figure.update_yaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            "Age is the single strongest predictor in the model. The spread at "
            "any given age is what the other features exist to explain."
        )

    with tab3:
        feature = st.selectbox(
            "Group by",
            options=config.CATEGORICAL_FEATURES,
            format_func=lambda c: config.FEATURE_LABELS.get(c, c),
        )
        counts = view[feature].value_counts()
        # Too many model families to plot them all legibly.
        keep = counts.head(20).index
        subset = view[view[feature].isin(keep)]
        figure = px.box(
            subset,
            x=feature,
            y=config.TARGET,
            color_discrete_sequence=[sup.ACCENT],
            labels={
                feature: config.FEATURE_LABELS.get(feature, feature),
                config.TARGET: "Selling price (INR)",
            },
        )
        figure.update_layout(
            template=sup.PLOT_TEMPLATE, height=460,
            title=f"Price by {config.FEATURE_LABELS.get(feature, feature).lower()}"
                  + (" (top 20 by count)" if len(counts) > 20 else ""),
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis={"categoryorder": "median descending"},
        )
        figure.update_yaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(figure, use_container_width=True)

        st.markdown("**Listing counts** — categories with few rows give weak predictions.")
        st.dataframe(
            counts.rename("listings").to_frame().assign(
                share=lambda d: (100 * d["listings"] / len(view)).round(1)
            ),
            use_container_width=True,
            column_config={
                "listings": st.column_config.NumberColumn("Listings", format="%d"),
                "share": st.column_config.NumberColumn("Share of view", format="%.1f%%"),
            },
        )

    with tab4:
        st.dataframe(
            view[["name", "year", "km_driven", "fuel", "transmission", "owner",
                  "max_power_bhp", "engine_cc", config.TARGET]]
            .sort_values(config.TARGET, ascending=False),
            use_container_width=True,
            height=420,
            column_config={
                config.TARGET: st.column_config.NumberColumn(
                    "Selling price (INR)", format="%d"
                ),
                "km_driven": st.column_config.NumberColumn("km driven", format="%d"),
            },
        )

    # ----------------------------------------------------- cleaning provenance
    exclusions = sup.get_exclusion_log()
    if exclusions:
        with st.expander("How this dataset was cleaned"):
            st.markdown(
                f"Started from **{exclusions['rows_in']:,}** raw rows and kept "
                f"**{exclusions['rows_out']:,}** "
                f"({exclusions['pct_removed']}% removed)."
            )
            applied = [s for s in exclusions["steps"] if s["rows_removed"] > 0]
            st.dataframe(
                [
                    {
                        "Reason": s["reason"],
                        "Rows removed": s["rows_removed"],
                        "Detail": s["detail"],
                    }
                    for s in applied
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(
                "Only duplicated and physically impossible records were "
                "removed. Expensive, very old and very high-mileage cars were "
                "deliberately kept — dropping awkward examples would improve "
                "the reported metrics without improving the model."
            )
