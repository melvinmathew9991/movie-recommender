"""
split.py

Per-user chronological train / validation / test splitting.

WHY THIS REPLACED THE GLOBAL SPLIT
----------------------------------
The original split cut the whole dataset at the 80th percentile of timestamp,
then dropped any test row whose user or movie was unseen in training. On the
synthetic sample that cost nothing -- every user was active across the whole time
range. On real MovieLens it discarded 86% of the test set and left 95 of 943
users, because real users *arrive over time*: the last 20% of ratings is
dominated by accounts that did not exist during the training window.

Holding out each user's own most recent ratings keeps every user in the
evaluation while still never training on a rating that post-dates one being
predicted for that user.

THE TRADE-OFF, STATED PLAINLY
-----------------------------
This is leakage-free *within* a user, but not globally time-ordered: user A's
training ratings may post-date user B's test ratings. A model could in principle
exploit population-level information "from the future".

That is accepted practice in the recommender literature, and MovieLens ships
its own splits (ua.base / ua.test) built exactly this way. The alternative --
strict global ordering -- costs 89% of the evaluation set, which buys temporal
purity at the price of a meaningless sample size.

WHY A FIXED HOLDOUT AND NOT A PERCENTAGE
----------------------------------------
Holding out a *fraction* per user makes Precision@K incomparable between users.
MovieLens guarantees only 20 ratings per user, so a 10% holdout gives that user
2 test items -- and Precision@10 then cannot exceed 0.2 no matter how good the
model is. The ceiling would be set by the split, not the model.

A fixed holdout gives every evaluated user the same number of held-out items, so
Precision@K means one consistent thing across the population.
"""

import pandas as pd

# Held out per user, most recent first: N_TEST most recent, then N_VAL before those.
N_TEST = 10
N_VAL = 10

# A user needs enough history to give up N_TEST + N_VAL and still have a
# meaningful amount left to learn from. At 30, the minimum-history user keeps 10
# training ratings.
MIN_RATINGS_FOR_EVAL = 30


def per_user_chronological_split(ratings: pd.DataFrame,
                                 n_test: int = N_TEST,
                                 n_val: int = N_VAL,
                                 min_ratings: int = MIN_RATINGS_FOR_EVAL):
    """
    Split into (train, validation, test) by holding out each eligible user's most
    recent ratings.

    Users with fewer than `min_ratings` ratings contribute **all** of their data
    to training and appear in neither evaluation set. They are not discarded --
    their ratings still inform the item factors.

    Returns (train_df, val_df, test_df).
    """
    ratings = ratings.sort_values(["user_id", "timestamp"], kind="mergesort")

    counts = ratings.groupby("user_id")["rating"].transform("size")
    eligible = counts >= min_ratings

    # Rank each rating within its user, most recent = 0
    recency = (ratings.groupby("user_id")["timestamp"]
               .rank(method="first", ascending=False) - 1)

    is_test = eligible & (recency < n_test)
    is_val = eligible & (recency >= n_test) & (recency < n_test + n_val)

    test_df = ratings[is_test].copy()
    val_df = ratings[is_val].copy()
    train_df = ratings[~(is_test | is_val)].copy()

    # An evaluation row is only usable if its user and movie were both seen in
    # training -- otherwise we are measuring cold start, a separate problem.
    train_users = set(train_df["user_id"])
    train_movies = set(train_df["movie_id"])

    def keep_known(df):
        return df[df["user_id"].isin(train_users) & df["movie_id"].isin(train_movies)].copy()

    return train_df, keep_known(val_df), keep_known(test_df)


def split_summary(ratings, train_df, val_df, test_df) -> str:
    """Human-readable description of what the split produced."""
    total_users = ratings["user_id"].nunique()
    lines = [
        f"  Train:      {len(train_df):>6,} ratings | {train_df['user_id'].nunique():>4} users",
        f"  Validation: {len(val_df):>6,} ratings | {val_df['user_id'].nunique():>4} users",
        f"  Test:       {len(test_df):>6,} ratings | {test_df['user_id'].nunique():>4} users",
        f"  Users in the catalog: {total_users} "
        f"({test_df['user_id'].nunique() / total_users:.0%} evaluated)",
    ]
    return "\n".join(lines)


def build_seen_dict(train_df) -> dict:
    """movie_ids each user already rated in training, for exclude_seen."""
    return train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
