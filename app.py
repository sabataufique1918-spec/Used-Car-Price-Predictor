"""Used Car Price Predictor - Streamlit entry point.

Run with:
    streamlit run app.py

This file only wires up navigation and shared chrome. It never trains a model:
it loads the artefacts that ``scripts/train_model.py`` produced. If those are
absent it says so and prints the commands to generate them.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project's own package importable before anything else imports it.
# Streamlit runs this file as a script, so src/ is not on sys.path by default,
# and on Streamlit Community Cloud the working directory is not guaranteed.
PROJECT_ROOT = Path(__file__).resolve().parent
for extra in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import streamlit as st  # noqa: E402

from ucpp import app_support as sup  # noqa: E402
from ucpp import config  # noqa: E402

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon=":material/directions_car:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Predict Price": "predict",
    "Data Explorer": "explorer",
    "Model Evaluation": "evaluation",
    "About": "about",
}


def main() -> None:
    with st.sidebar:
        st.title("Used Car Price Predictor")
        st.caption(
            f"Historical Indian used-car listings, model years up to "
            f"{config.OBSERVATION_YEAR}"
        )
        choice = st.radio("Page", list(PAGES), label_visibility="collapsed")
        st.divider()

    # Every page except About needs the trained artefacts.
    if choice != "About" and not sup.artefacts_available():
        st.title(choice)
        sup.render_missing_artefacts()
        return

    module_name = PAGES[choice]
    # Imported lazily so a page's own imports (and its Plotly figures) are only
    # loaded when that page is actually viewed.
    page = __import__(f"app_pages.{module_name}", fromlist=["render"])
    page.render()

    with st.sidebar:
        report = sup.get_evaluation_report()
        if report:
            test = report["test_metrics"]
            st.caption("**Held-out test performance**")
            st.caption(
                f"MAE {sup.format_inr(test['mae_inr'])}  \n"
                f"R² {test['r2']:.3f}  \n"
                f"Median error {test['median_abs_pct_error']:.1f}%"
            )
            st.divider()
        st.caption(
            "Estimates reflect historical listing prices, not a current or "
            "verified market valuation."
        )


if __name__ == "__main__":
    main()
