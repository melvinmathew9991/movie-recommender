"""
streamlit_app.py -- full-stack demo frontend.

Calls the FastAPI service (not the model directly) to keep the same
API-first architecture used in the Nectar and MEDBOT projects -- the
frontend is a client of the API, not a shortcut around it.

Run with `streamlit run app/streamlit_app.py`; requires the API
(api/main.py) to be running first.

Verified end-to-end: the script is executed against a live API in
tests/test_frontend.py using Streamlit's own AppTest harness, which runs
the real script body and asserts on the rendered widgets -- including that
the recommendation table preserves the model's ranking order.
"""

import os

import streamlit as st
import requests
import pandas as pd
from pathlib import Path

# Read from the environment so this works both locally (API on localhost) and
# under docker-compose, where the API is a separate service reachable at
# http://api:8000 -- inside the frontend container, localhost is the frontend
# itself, so a hardcoded value can never reach the API.
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommender")
st.caption("Full-stack demo -- Streamlit frontend calling a FastAPI service backed by a "
           "from-scratch matrix factorization model.")

movies_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "movies.csv"
movies_df = pd.read_csv(movies_path) if movies_path.exists() else None

user_id = st.number_input("User ID", min_value=1, value=1, step=1)
k = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

if st.button("Get Recommendations"):
    try:
        resp = requests.get(f"{API_BASE}/recommend/{user_id}", params={"k": k}, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        st.success(f"Recommendations for user {data['user_id']} (model: {data['model_version']})")

        rec_ids = data["recommended_movie_ids"]
        if movies_df is not None:
            # Join by reindexing on movie_id rather than filtering with `isin`:
            # a boolean filter returns rows in catalog order, which silently
            # discards the model's ranking -- and the ranking is the entire
            # point of a top-K list. Reindexing keeps row i as the i-th
            # recommendation.
            ranked = movies_df.set_index("movie_id").reindex(rec_ids).reset_index()
            ranked.insert(0, "rank", range(1, len(ranked) + 1))

            unknown = int(ranked["title"].isna().sum())
            if unknown:
                st.warning(f"{unknown} recommended movie ID(s) are not present in "
                           "the local catalog file and will show as blank rows.")

            st.table(ranked[["rank", "movie_id", "title", "genre", "release_year"]])
        else:
            st.write(rec_ids)

    except requests.exceptions.ConnectionError:
        st.error("Could not reach the API. Is it running? "
                 "Start it with: uvicorn api.main:app --reload")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")

st.divider()
st.caption("API health check:")
try:
    health = requests.get(f"{API_BASE}/health", timeout=2).json()
    st.json(health)
except requests.exceptions.RequestException:
    st.warning("API not reachable.")
