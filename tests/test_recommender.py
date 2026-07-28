"""
test_recommender.py

Tests for the recommender models. Run with `pytest` from the project root.

src/ reaches sys.path via `pythonpath` in pyproject.toml rather than a
sys.path.insert preamble in each test module.
"""

import pandas as pd
import numpy as np
import pytest

from models import (PopularityRecommender, MatrixFactorizationRecommender,
                    RandomRecommender)
from evaluate import rmse, mae, precision_at_k, recall_at_k
from audit_bias import audit_popularity_bias, compare_bias


def _toy_ratings_df():
    """Small, deterministic ratings dataframe for fast, repeatable tests."""
    data = {
        "user_id":   [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        "movie_id":  [1, 2, 3, 1, 2, 4, 1, 3, 4, 5],
        "rating":    [5, 3, 4, 4, 5, 2, 3, 5, 4, 1],
        "timestamp": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    }
    return pd.DataFrame(data)


# --- Metric correctness -------------------------------------------------------


def test_rmse_zero_for_perfect_predictions():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_rmse_known_value():
    # errors of 1, 1 -> mean squared error 1 -> rmse 1
    assert abs(rmse([1, 2], [2, 3]) - 1.0) < 1e-9


def test_mae_known_value():
    assert abs(mae([1, 2, 3], [2, 2, 5]) - 1.0) < 1e-9


def test_precision_at_k_all_relevant():
    assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0


def test_precision_at_k_none_relevant():
    assert precision_at_k([1, 2, 3], {7, 8, 9}, k=3) == 0.0


def test_precision_at_k_partial():
    # top-2 of [1,2,3,4] = [1,2]; only 1 is relevant -> 1/2
    assert abs(precision_at_k([1, 2, 3, 4], {1, 3}, k=2) - 0.5) < 1e-9


def test_recall_at_k():
    # 3 of 4 relevant items appear in the top-3 -> 0.75
    assert abs(recall_at_k([1, 2, 3], {1, 2, 3, 4}, k=3) - 0.75) < 1e-9


def test_recall_at_k_no_relevant_items_is_zero_not_a_crash():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


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


# --- Bias audit ---------------------------------------------------------------


def test_audit_bias_reports_coverage_in_valid_range():
    df = _toy_ratings_df()
    model = PopularityRecommender(min_ratings=1).fit(df)
    result = audit_popularity_bias(
        model, users=[1, 2, 3], all_movie_ids=[1, 2, 3, 4, 5],
        train_seen_by_user={}, k=3
    )
    assert "catalog_coverage" in result
    assert 0.0 <= result["catalog_coverage"] <= 1.0
    assert 0.0 <= result["top_item_share"] <= 1.0


def test_audit_bias_handles_a_model_that_recommends_nothing():
    """Every candidate excluded -> no recommendations -> must not divide by zero."""
    df = _toy_ratings_df()
    model = PopularityRecommender(min_ratings=1).fit(df)
    result = audit_popularity_bias(
        model, users=[1], all_movie_ids=[1, 2, 3, 4, 5],
        train_seen_by_user={1: {1, 2, 3, 4, 5}}, k=3
    )
    assert result["n_recommendations"] == 0
    assert result["catalog_coverage"] == 0.0


# --- Random reference baseline ------------------------------------------------


def test_random_recommender_is_reproducible():
    df = _toy_ratings_df()
    a = RandomRecommender(random_state=7).fit(df).recommend(1, k=3)
    b = RandomRecommender(random_state=7).fit(df).recommend(1, k=3)
    assert a == b, "Same seed should give the same reference picks"


def test_random_recommender_excludes_seen():
    df = _toy_ratings_df()
    all_movies = set(df["movie_id"])
    model = RandomRecommender(random_state=7).fit(df)
    assert model.recommend(1, k=5, exclude_seen=all_movies) == []


def test_compare_bias_table_columns_are_aligned():
    """
    The label column must fit the longest metric name. At 28 chars,
    'Top-item recommendation share' (29) overflowed and shifted that row.
    """
    pop = {"catalog_coverage": 0.05, "top_item_share": 0.096}
    mf = {"catalog_coverage": 0.63, "top_item_share": 0.046}
    rnd = {"catalog_coverage": 0.98, "top_item_share": 0.008}

    rows = [line for line in compare_bias(pop, mf, rnd).splitlines()
            if line.startswith(("Metric", "Catalog coverage", "Top-item"))]
    assert len(rows) == 3

    # longest label (29) + 1 char padding + 1 separator space
    value_col = len("Top-item recommendation share") + 2
    for row in rows:
        assert row[value_col - 1] == " ", f"Label column not padded: {row!r}"
        assert row[value_col] != " ", f"Value column misaligned: {row!r}"


def test_compare_bias_still_works_without_the_random_column():
    pop = {"catalog_coverage": 0.05, "top_item_share": 0.096}
    mf = {"catalog_coverage": 0.63, "top_item_share": 0.046}
    assert "Random" not in compare_bias(pop, mf)


# --- Input guards -------------------------------------------------------------


@pytest.mark.parametrize("model_factory", [
    lambda df: PopularityRecommender(min_ratings=1).fit(df),
    lambda df: MatrixFactorizationRecommender(n_factors=2, n_epochs=3, random_state=42).fit(df),
    lambda df: RandomRecommender(random_state=42).fit(df),
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
    (RandomRecommender, {}),
])
def test_recommend_before_fit_raises_runtime_error(model_class, kwargs):
    """An unfitted model should say so, not fail on a None attribute."""
    model = model_class(**kwargs)
    with pytest.raises(RuntimeError):
        model.recommend(user_id=1, k=5)
