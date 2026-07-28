"""
test_data_generation.py

Guards the sample data generator: correct schema, valid ranges, reproducible
from a seed -- and, most importantly, that the ratings actually depend on who
is rating and what is being rated.

That last group exists because the previous generator drew every rating from one
fixed distribution, so the dataset held no recoverable preference signal. The
models trained and scored fine while measuring nothing, and no other test in the
suite caught it.
"""

import csv

import numpy as np
import pytest

import generate_sample_data as gen


@pytest.fixture
def rng():
    return np.random.default_rng(gen.SEED)


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --- Schema and ranges --------------------------------------------------------


def test_movies_file_schema_and_ranges(tmp_path, rng):
    out = tmp_path / "movies.csv"
    gen.generate_movies(out, rng)
    rows = _read(out)

    assert len(rows) == gen.N_MOVIES
    assert list(rows[0].keys()) == ["movie_id", "title", "genre", "release_year"]
    assert {int(r["movie_id"]) for r in rows} == set(range(1, gen.N_MOVIES + 1))
    assert all(r["genre"] in gen.GENRES for r in rows)
    assert all(1990 <= int(r["release_year"]) <= 2020 for r in rows)


def test_ratings_file_schema_and_ranges(tmp_path, rng):
    out = tmp_path / "ratings.csv"
    gen.generate_ratings(out, rng)
    rows = _read(out)

    assert len(rows) == gen.N_RATINGS
    assert list(rows[0].keys()) == ["user_id", "movie_id", "rating", "timestamp"]
    assert all(1 <= int(r["rating"]) <= 5 for r in rows), "ratings must stay on the 1-5 scale"
    assert all(1 <= int(r["user_id"]) <= gen.N_USERS for r in rows)
    assert all(1 <= int(r["movie_id"]) <= gen.N_MOVIES for r in rows)


def test_a_user_never_rates_the_same_movie_twice(tmp_path, rng):
    out = tmp_path / "ratings.csv"
    gen.generate_ratings(out, rng)
    rows = _read(out)

    pairs = {(r["user_id"], r["movie_id"]) for r in rows}
    assert len(pairs) == len(rows), "duplicate (user, movie) pairs would corrupt the train/test split"


def test_generation_is_reproducible_from_the_same_seed(tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    gen.generate_ratings(first, np.random.default_rng(gen.SEED))
    gen.generate_ratings(second, np.random.default_rng(gen.SEED))
    assert first.read_bytes() == second.read_bytes(), "same seed must reproduce the dataset"


# --- Preference signal (regression tests for the no-signal defect) -------------


def test_ratings_vary_by_user_and_by_movie(rng):
    """
    Regression: ratings were drawn independently of user and movie, leaving
    nothing for a recommender to learn. The generative model's true scores must
    vary across users for a fixed movie (taste differs between people) and
    across movies for a fixed user (quality differs between titles).
    """
    taste, attrs, user_bias, quality = gen._latent_factors(rng)

    across_users = np.array([
        gen._true_score(u, 42, taste, attrs, user_bias, quality)
        for u in range(1, gen.N_USERS + 1)
    ])
    across_movies = np.array([
        gen._true_score(7, i, taste, attrs, user_bias, quality)
        for i in range(1, gen.N_MOVIES + 1)
    ])

    assert across_users.std() > 0.2, (
        f"Ratings barely vary between users (std={across_users.std():.3f}); "
        "there is no personalization signal to learn."
    )
    assert across_movies.std() > 0.2, (
        f"Ratings barely vary between movies (std={across_movies.std():.3f})."
    )


def test_users_disagree_about_which_movies_are_better(rng):
    """
    The interaction term must actually matter. If ratings were only
    user_bias + movie_quality, every user would rank all movies identically and
    "personalization" would just be a relabelled global ranking.
    """
    taste, attrs, user_bias, quality = gen._latent_factors(rng)

    def prefers(u, i, j):
        return (gen._true_score(u, i, taste, attrs, user_bias, quality)
                > gen._true_score(u, j, taste, attrs, user_bias, quality))

    disagreements = sum(
        1 for i in range(1, 40) for j in range(i + 1, 40)
        if prefers(1, i, j) != prefers(2, i, j)
    )
    assert disagreements > 0, "All users rank every movie pair identically -- no personalization."


def test_generated_ratings_carry_more_user_signal_than_chance(tmp_path, rng):
    """
    The end-to-end version of the permutation test that found the original
    defect: per-user mean-rating spread must exceed what the same ratings give
    when shuffled at random. On the old generator this was NEGATIVE.
    """
    out = tmp_path / "ratings.csv"
    gen.generate_ratings(out, rng)
    rows = _read(out)

    users = np.array([int(r["user_id"]) for r in rows])
    ratings = np.array([int(r["rating"]) for r in rows])

    def user_spread(vals):
        return float(np.std([vals[users == u].mean() for u in np.unique(users)]))

    observed = user_spread(ratings)

    shuffler = np.random.default_rng(0)
    null = np.array([user_spread(shuffler.permutation(ratings)) for _ in range(30)])
    z = (observed - null.mean()) / null.std()

    assert z > 5, (
        f"Per-user rating spread is indistinguishable from chance (z={z:.2f}). "
        "The dataset has no user preference signal."
    )
