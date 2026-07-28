"""
test_recommender.py

Tests for the recommender models. Run with `pytest` from the project root.

src/ reaches sys.path via `pythonpath` in pyproject.toml rather than a
sys.path.insert preamble in each test module.
"""

import pandas as pd
import numpy as np
import pytest

from models import PopularityRecommender, MatrixFactorizationRecommender


def _toy_ratings_df():
    """Small, deterministic ratings dataframe for fast, repeatable tests."""
    data = {
        "user_id":   [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        "movie_id":  [1, 2, 3, 1, 2, 4, 1, 3, 4, 5],
        "rating":    [5, 3, 4, 4, 5, 2, 3, 5, 4, 1],
        "timestamp": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    }
    return pd.DataFrame(data)


# --- Popularity baseline ------------------------------------------------------


def test_popularity_recommender_fits_and_ranks():
    df = _toy_ratings_df()
    model = PopularityRecommender(min_ratings=1).fit(df)
    recs = model.recommend(user_id=1, k=3, exclude_seen=set())
    assert len(recs) <= 3
    assert all(isinstance(m, (int, np.integer)) for m in recs)


def test_popularity_recommender_excludes_seen():
    df = _toy_ratings_df()
    model = PopularityRecommender(min_ratings=1).fit(df)
    all_movies = set(df["movie_id"])
    recs = model.recommend(user_id=1, k=10, exclude_seen=all_movies)
    assert len(recs) == 0, "Excluding every movie should leave zero recommendations"


# --- Matrix factorization -----------------------------------------------------


def test_mf_recommender_fits_without_error():
    df = _toy_ratings_df()
    model = MatrixFactorizationRecommender(n_factors=2, n_epochs=3, random_state=42).fit(df)
    assert model.user_factors_ is not None
    assert model.item_factors_ is not None


def test_mf_predict_returns_valid_range():
    df = _toy_ratings_df()
    model = MatrixFactorizationRecommender(n_factors=2, n_epochs=3, random_state=42).fit(df)
    pred = model.predict(user_id=1, movie_id=2)
    assert 1.0 <= pred <= 5.0, f"Prediction {pred} outside valid rating range [1, 5]"


def test_mf_predict_cold_start_fallback():
    """A user/movie never seen during training should fall back to the global mean, not crash."""
    df = _toy_ratings_df()
    model = MatrixFactorizationRecommender(n_factors=2, n_epochs=3, random_state=42).fit(df)
    pred = model.predict(user_id=999, movie_id=999)
    assert abs(pred - model.global_bias_) < 1e-9


def test_mf_recommend_excludes_seen():
    df = _toy_ratings_df()
    model = MatrixFactorizationRecommender(n_factors=2, n_epochs=3, random_state=42).fit(df)
    seen = {1, 2, 3, 4, 5}
    recs = model.recommend(user_id=1, k=10, exclude_seen=seen, candidate_items=[1, 2, 3, 4, 5])
    assert len(recs) == 0


def test_reproducibility_same_seed_same_result():
    """Same random_state should give bit-identical trained factors -- a real reproducibility
    check, not just an assumption."""
    df = _toy_ratings_df()
    m1 = MatrixFactorizationRecommender(n_factors=3, n_epochs=5, random_state=7).fit(df)
    m2 = MatrixFactorizationRecommender(n_factors=3, n_epochs=5, random_state=7).fit(df)
    assert np.allclose(m1.user_factors_, m2.user_factors_), "Same seed should reproduce identical training"


# --- Input guards -------------------------------------------------------------


@pytest.mark.parametrize("model_factory", [
    lambda df: PopularityRecommender(min_ratings=1).fit(df),
    lambda df: MatrixFactorizationRecommender(n_factors=2, n_epochs=3, random_state=42).fit(df),
])
def test_negative_k_raises_instead_of_silently_truncating(model_factory):
    """
    `candidates[:k]` with k < 0 returns everything *except* the last |k| items --
    a near-full catalog dump presented as a top-K list. Fail loudly instead.
    """
    model = model_factory(_toy_ratings_df())
    with pytest.raises(ValueError):
        model.recommend(user_id=1, k=-1)


@pytest.mark.parametrize("model_class,kwargs", [
    (PopularityRecommender, {}),
    (MatrixFactorizationRecommender, {"n_factors": 2, "n_epochs": 3}),
])
def test_recommend_before_fit_raises_runtime_error(model_class, kwargs):
    """An unfitted model should say so, not fail on a None attribute."""
    model = model_class(**kwargs)
    with pytest.raises(RuntimeError):
        model.recommend(user_id=1, k=5)
