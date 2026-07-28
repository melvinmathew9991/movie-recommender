"""
test_split.py

Tests for the per-user chronological split.

The properties that matter are leakage-safety and coverage. The split this
replaced was leakage-safe but evaluated only 10% of users on real data; a split
that keeps everyone is worthless if it achieves that by leaking.
"""

import pandas as pd
import pytest

from split import per_user_chronological_split, build_seen_dict


def _ratings(n_users=5, n_per_user=40, n_movies=15):
    """
    Users each rating a run of movies at strictly increasing timestamps.

    The catalog is deliberately small relative to the holdout. Every user's
    training portion then covers the whole catalog, so no evaluation row is
    dropped by the seen-in-training filter and these tests measure the split
    logic rather than catalog coverage. (With a large sparse catalog, a user's
    held-out movies can be absent from every training set and that user vanishes
    from the evaluation -- real behaviour, but not what is under test here.)
    """
    rows = []
    for u in range(1, n_users + 1):
        for j in range(n_per_user):
            rows.append({
                "user_id": u,
                "movie_id": (u * 3 + j) % n_movies + 1,
                "rating": (j % 5) + 1,
                # Interleave users in time so the split cannot rely on user order
                "timestamp": 1_000_000 + j * 10 + u,
            })
    return pd.DataFrame(rows)


# --- Leakage ------------------------------------------------------------------


def test_no_training_rating_postdates_that_users_test_rating():
    """The core guarantee: within a user, training is strictly in the past."""
    df = _ratings()
    train, val, test = per_user_chronological_split(df)

    for user in test["user_id"].unique():
        latest_train = train.loc[train["user_id"] == user, "timestamp"].max()
        earliest_test = test.loc[test["user_id"] == user, "timestamp"].min()
        assert latest_train < earliest_test, f"user {user}: training leaks past test"


def test_validation_sits_between_train_and_test_in_time():
    df = _ratings()
    train, val, test = per_user_chronological_split(df)

    for user in val["user_id"].unique():
        latest_train = train.loc[train["user_id"] == user, "timestamp"].max()
        val_times = val.loc[val["user_id"] == user, "timestamp"]
        earliest_test = test.loc[test["user_id"] == user, "timestamp"].min()
        assert latest_train < val_times.min()
        assert val_times.max() < earliest_test


def test_splits_are_disjoint():
    df = _ratings()
    train, val, test = per_user_chronological_split(df)

    def keys(d):
        return set(zip(d["user_id"], d["movie_id"], d["timestamp"]))

    assert not keys(train) & keys(val)
    assert not keys(train) & keys(test)
    assert not keys(val) & keys(test)


# --- Coverage -----------------------------------------------------------------


def test_every_eligible_user_is_evaluated():
    """
    The whole point of this split. The previous global-timestamp split left
    95 of 943 real users in the test set; this must keep all eligible ones.
    """
    df = _ratings(n_users=5, n_per_user=40)
    train, val, test = per_user_chronological_split(df)
    assert test["user_id"].nunique() == 5
    assert val["user_id"].nunique() == 5


def test_holdout_size_is_fixed_per_user_not_proportional():
    """
    A fixed holdout keeps Precision@K comparable across users. A percentage
    holdout would cap a 20-rating user's Precision@10 at 0.2.
    """
    df = pd.concat([_ratings(n_users=1, n_per_user=40),
                    _ratings(n_users=1, n_per_user=100).assign(user_id=2)])
    _, val, test = per_user_chronological_split(df)

    assert test.groupby("user_id").size().nunique() == 1, "test holdout must be uniform"
    assert val.groupby("user_id").size().nunique() == 1, "val holdout must be uniform"


def test_low_history_users_train_the_model_but_are_not_evaluated():
    """
    Users below the threshold cannot give up 20 ratings and still be learnable.
    They are excluded from evaluation but their ratings still inform item factors.
    """
    sparse = _ratings(n_users=1, n_per_user=12)          # below the 30 threshold
    dense = _ratings(n_users=1, n_per_user=40).assign(user_id=2)
    df = pd.concat([sparse, dense])

    train, val, test = per_user_chronological_split(df)

    assert 1 not in set(test["user_id"]), "sparse user must not be evaluated"
    assert 1 not in set(val["user_id"])
    assert (train["user_id"] == 1).sum() == 12, "but all their ratings must train"


# --- Evaluation rows must be predictable --------------------------------------


def test_evaluation_rows_only_reference_users_and_movies_seen_in_training():
    df = _ratings()
    train, val, test = per_user_chronological_split(df)

    train_users, train_movies = set(train["user_id"]), set(train["movie_id"])
    for name, part in (("val", val), ("test", test)):
        assert set(part["user_id"]) <= train_users, f"{name} has unseen users"
        assert set(part["movie_id"]) <= train_movies, f"{name} has unseen movies"


def test_build_seen_dict_maps_users_to_their_training_movies():
    df = _ratings(n_users=2, n_per_user=40)
    train, _, _ = per_user_chronological_split(df)
    seen = build_seen_dict(train)

    assert set(seen) == set(train["user_id"])
    for user, movies in seen.items():
        assert movies == set(train.loc[train["user_id"] == user, "movie_id"])
