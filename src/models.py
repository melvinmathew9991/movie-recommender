"""
models.py

Three recommenders, deliberately contrasted:

1. PopularityRecommender -- the naive baseline every recommender project
   needs. Recommends whatever's most-rated/highest-rated globally, with
   zero personalization. This exists specifically so the bias audit has
   something to compare against.

2. RandomRecommender -- the floor. Not a model, a reference point: it is
   what "learned nothing" scores. Without it, metrics like catalog
   coverage have no scale to be read against.

3. MatrixFactorizationRecommender -- a real collaborative-filtering model,
   implemented from scratch via stochastic gradient descent (not calling
   scikit-surprise or implicit), following the standard formulation:

       predicted_rating(u, i) = global_bias + user_bias[u] + item_bias[i]
                                 + dot(user_factors[u], item_factors[i])

   trained by minimizing squared error + L2 regularization via SGD,
   which is the same core algorithm behind the Netflix Prize-winning
   approaches (Funk MF / biased SVD), just written out explicitly rather
   than imported.
"""

import numpy as np


class PopularityRecommender:
    """Naive baseline: recommend by global mean rating among sufficiently-rated items."""

    def __init__(self, min_ratings: int = 5):
        self.min_ratings = min_ratings
        self.item_scores_ = None  # movie_id -> mean rating
        self.item_counts_ = None

    def fit(self, ratings_df):
        grp = ratings_df.groupby("movie_id")["rating"]
        self.item_counts_ = grp.count()
        means = grp.mean()
        # Only rank items with enough ratings to be meaningful
        eligible = self.item_counts_[self.item_counts_ >= self.min_ratings].index
        self.item_scores_ = means.loc[eligible].sort_values(ascending=False)
        return self

    def recommend(self, user_id, k: int = 10, exclude_seen: set = None):
        if k < 0:
            # Without this, `candidates[:k]` quietly returns everything *except*
            # the last |k| items instead of failing -- a wrong answer, not an error.
            raise ValueError(f"k must be non-negative, got {k}")
        if self.item_scores_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        exclude_seen = exclude_seen or set()
        candidates = [m for m in self.item_scores_.index if m not in exclude_seen]
        return candidates[:k]


class RandomRecommender:
    """
    Uniformly random picks from the catalog. Not a serious recommender -- it is
    the reference point that makes the other numbers interpretable.

    Without it, "MF covers 63% of the catalog" reads as evidence of
    personalization, when random selection covers nearly everything: broad
    coverage on its own only rules out popularity collapse, it does not
    demonstrate per-user targeting. Precision@K against random is the check on
    whether the model has learned anything beyond chance.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.items_ = None

    def fit(self, ratings_df):
        self.items_ = ratings_df["movie_id"].unique()
        return self

    def recommend(self, user_id, k: int = 10, exclude_seen: set = None):
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if self.items_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        exclude_seen = exclude_seen or set()
        pool = [m for m in self.items_ if m not in exclude_seen]
        size = min(k, len(pool))
        if size == 0:
            return []
        return [pool[i] for i in self.rng.choice(len(pool), size=size, replace=False)]


class MatrixFactorizationRecommender:
    """
    Biased matrix factorization trained via SGD, implemented from raw
    NumPy operations -- no scikit-surprise, no implicit library.
    """

    def __init__(self, n_factors: int = 20, learning_rate: float = 0.01,
                 reg: float = 0.02, n_epochs: int = 30, random_state: int = 42):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.rng = np.random.default_rng(random_state)

        self.user_factors_ = None
        self.item_factors_ = None
        self.user_bias_ = None
        self.item_bias_ = None
        self.global_bias_ = None
        self.user_id_map_ = None
        self.item_id_map_ = None

    def fit(self, ratings_df, verbose: bool = False):
        users = ratings_df["user_id"].unique()
        items = ratings_df["movie_id"].unique()
        self.user_id_map_ = {u: idx for idx, u in enumerate(users)}
        self.item_id_map_ = {i: idx for idx, i in enumerate(items)}

        n_users, n_items = len(users), len(items)
        self.user_factors_ = self.rng.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors_ = self.rng.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias_ = np.zeros(n_users)
        self.item_bias_ = np.zeros(n_items)
        self.global_bias_ = ratings_df["rating"].mean()

        # Precompute index arrays once
        u_idx = ratings_df["user_id"].map(self.user_id_map_).values
        i_idx = ratings_df["movie_id"].map(self.item_id_map_).values
        r_vals = ratings_df["rating"].values.astype(float)

        n = len(r_vals)
        for epoch in range(self.n_epochs):
            perm = self.rng.permutation(n)
            total_sq_err = 0.0
            for idx in perm:
                u, i, r = u_idx[idx], i_idx[idx], r_vals[idx]
                pred = (self.global_bias_ + self.user_bias_[u] + self.item_bias_[i]
                        + np.dot(self.user_factors_[u], self.item_factors_[i]))
                err = r - pred
                total_sq_err += err ** 2

                # SGD updates
                self.user_bias_[u] += self.lr * (err - self.reg * self.user_bias_[u])
                self.item_bias_[i] += self.lr * (err - self.reg * self.item_bias_[i])

                uf_old = self.user_factors_[u].copy()
                self.user_factors_[u] += self.lr * (err * self.item_factors_[i] - self.reg * self.user_factors_[u])
                self.item_factors_[i] += self.lr * (err * uf_old - self.reg * self.item_factors_[i])

            if verbose:
                rmse = np.sqrt(total_sq_err / n)
                print(f"  epoch {epoch + 1}/{self.n_epochs}  train RMSE: {rmse:.4f}")

        return self

    def predict(self, user_id, movie_id) -> float:
        u = self.user_id_map_.get(user_id)
        i = self.item_id_map_.get(movie_id)
        if u is None or i is None:
            return self.global_bias_  # cold start fallback
        pred = (self.global_bias_ + self.user_bias_[u] + self.item_bias_[i]
                + np.dot(self.user_factors_[u], self.item_factors_[i]))
        return float(np.clip(pred, 1, 5))

    def recommend(self, user_id, k: int = 10, exclude_seen: set = None, candidate_items=None):
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if self.item_id_map_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        exclude_seen = exclude_seen or set()
        if candidate_items is None:
            candidate_items = list(self.item_id_map_.keys())
        candidates = [m for m in candidate_items if m not in exclude_seen]
        scored = [(m, self.predict(user_id, m)) for m in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:k]]
