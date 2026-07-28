"""
evaluate.py

Two evaluation families, because they measure different things and a
recommender project needs both:

1. Rating-prediction accuracy (RMSE, MAE) -- "how close was the predicted
   star rating to the actual one?"

2. Ranking quality (Precision@K, Recall@K) -- "of the top-K items we'd
   actually show the user, how many were things they liked?" This is the
   metric that matters operationally -- a model can have mediocre RMSE
   but still produce a genuinely useful top-10 list, or vice versa.
"""

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Of the top-k recommended items, what fraction are in the relevant set?
    `relevant` = items the user rated >= 4 in the held-out test set.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Of all relevant items, what fraction appeared in the top-k recommendations?"""
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def evaluate_ranking(model, test_df, train_seen_by_user: dict, k: int = 10,
                      relevance_threshold: int = 4, max_users: int = None) -> dict:
    """
    Evaluates Precision@K / Recall@K across all (or a sample of) users
    who have at least one relevant item in the test set.
    """
    users = test_df["user_id"].unique()
    if max_users:
        users = users[:max_users]

    precisions, recalls = [], []
    for user_id in users:
        user_test = test_df[test_df["user_id"] == user_id]
        relevant = set(user_test.loc[user_test["rating"] >= relevance_threshold, "movie_id"])
        if not relevant:
            continue

        seen = train_seen_by_user.get(user_id, set())
        recs = model.recommend(user_id, k=k, exclude_seen=seen)

        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))

    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "n_users_evaluated": len(precisions),
        "k": k,
    }
