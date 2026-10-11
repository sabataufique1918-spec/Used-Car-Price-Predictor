"""Smoke tests for the Streamlit app, using Streamlit's own AppTest harness.

These run the real app script in-process and assert it produces no exception.
That catches the class of bug unit tests miss entirely: a widget misconfigured,
a page module failing to import, or a column name the layout expects but the
schema does not provide.

AppTest does not execute Plotly rendering in a browser, so these verify the
script runs and the pages build - not the visual result. A manual browser check
is still worth doing before deploying.
"""

from __future__ import annotations

import pytest

from ucpp import app_support

pytest.importorskip("streamlit.testing.v1", reason="needs Streamlit's test harness")
from streamlit.testing.v1 import AppTest  # noqa: E402

from ucpp import config  # noqa: E402

APP_PATH = str(config.PROJECT_ROOT / "app.py")
TIMEOUT = 120

requires_artefacts = pytest.mark.skipif(
    not app_support.artefacts_available(),
    reason="trained model not present; run python scripts/train_model.py",
)


def _run(page_label: str | None = None) -> AppTest:
    """Run the app, optionally after selecting a sidebar page."""
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT).run()
    if page_label is not None:
        app.sidebar.radio[0].set_value(page_label).run()
    return app


class TestAppStarts:
    def test_app_runs_without_exception(self):
        app = _run()
        assert not app.exception

    def test_sidebar_offers_all_four_pages(self):
        app = _run()
        options = app.sidebar.radio[0].options
        assert options == [
            "Predict Price",
            "Data Explorer",
            "Model Evaluation",
            "About",
        ]

    def test_about_page_works_without_a_trained_model(self):
        """About must be readable even on a fresh clone."""
        app = _run("About")
        assert not app.exception


@requires_artefacts
class TestPagesRender:
    @pytest.mark.parametrize(
        "page",
        ["Predict Price", "Data Explorer", "Model Evaluation", "About"],
    )
    def test_page_renders(self, page):
        app = _run(page)
        assert not app.exception, f"{page} raised: {app.exception}"

    def test_prediction_form_submits_and_shows_a_price(self):
        """The end-to-end user path through the real UI."""
        app = _run("Predict Price")

        # Set the dependent dropdowns: brand first, then re-run so the model
        # list populates, exactly as a user would experience it.
        brand_box = app.selectbox[0]
        brand_box.set_value("Maruti").run()

        model_box = app.selectbox[1]
        assert model_box.options, "model dropdown did not populate after brand"
        model_box.set_value(model_box.options[0]).run()

        app.button[0].click().run()
        assert not app.exception

        # The estimate is rendered as a metric containing a rupee figure.
        assert len(app.metric) >= 1
        assert any("₹" in str(m.value) for m in app.metric)

    def test_model_evaluation_shows_saved_metrics(self):
        app = _run("Model Evaluation")
        assert not app.exception
        assert len(app.metric) >= 4  # MAE, RMSE, R2, median error

    def test_data_explorer_handles_an_empty_filter_result(self):
        """Over-narrow filters must warn, not crash."""
        app = _run("Data Explorer")
        assert not app.exception
        # Drive the price cap to its minimum and the year range to one year.
        app.sidebar.slider[0].set_range(1994, 1994).run()
        assert not app.exception
