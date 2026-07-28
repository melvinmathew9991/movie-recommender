"""
test_frontend.py

Executes the Streamlit app with Streamlit's own AppTest harness. This runs the
real script body -- widgets, callbacks, rendering -- so the frontend is covered
by CI rather than only ever being eyeballed in a browser.

The API is stubbed at the `requests` layer. That keeps the test hermetic (no
live server, no port binding) while still exercising everything the app itself
is responsible for: reading API_BASE, issuing the request, and turning the
response into a rendered table.
"""

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
MOVIES_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "movies.csv"

# Deliberately NOT in ascending movie_id order -- ascending order is exactly what
# the old `isin` filter returned by accident, so it could not distinguish a
# correctly ranked table from a broken one.
RANKED_IDS = [59, 17, 76, 68, 225]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _fake_get(url, **kwargs):
    if "/health" in url:
        return _FakeResponse({"status": "ok", "model_loaded": True})
    if "/recommend/" in url:
        return _FakeResponse({
            "user_id": 1,
            "recommended_movie_ids": RANKED_IDS,
            "model_version": "mf-v1",
        })
    raise AssertionError(f"Unexpected URL requested by the app: {url}")


@pytest.fixture
def stubbed_api(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", _fake_get)


def test_app_renders_without_exception(stubbed_api):
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception, f"App raised on first render: {at.exception}"
    assert at.title[0].value == "🎬 Movie Recommender"
    assert at.button[0].label == "Get Recommendations"


def test_recommendation_flow_succeeds(stubbed_api):
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.button[0].click().run()

    assert not at.exception, f"App raised after requesting recommendations: {at.exception}"
    assert not at.error, f"App surfaced an error: {[e.value for e in at.error]}"
    assert at.success, "Expected a success message after a good API response"


@pytest.mark.skipif(not MOVIES_CSV.exists(),
                    reason="data/raw/movies.csv absent; run src/generate_sample_data.py first")
def test_table_preserves_model_ranking(stubbed_api):
    """
    Regression: the table was built with `movies_df[movies_df.movie_id.isin(ids)]`,
    which returns rows in catalog order. That silently re-sorted the top-K list by
    movie_id, so the highest-scoring recommendation was not shown first -- the one
    thing a ranked recommender must get right.
    """
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.button[0].click().run()

    tables = at.get("table")
    assert tables, "Expected a rendered recommendations table"

    rendered = tables[0].value
    assert list(rendered["movie_id"]) == RANKED_IDS, (
        "Table rows are not in the model's ranked order; "
        f"got {list(rendered['movie_id'])}, expected {RANKED_IDS}"
    )
    assert list(rendered["rank"]) == [1, 2, 3, 4, 5]
