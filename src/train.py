"""
train.py

End-to-end training pipeline:
1. Load data, split chronologically by timestamp (not randomly -- a
   random split would leak future ratings into training).
2. Train the popularity baseline and the from-scratch matrix
   factorization model.
3. Evaluate on RMSE/MAE (rating prediction) and Precision@K/Recall@K
   (ranking quality).
4. Run the popularity-bias audit comparing the two models.
5. Persist the trained MF model (pickle) for the API to load.
"""

import pickle
import pandas as pd
from pathlib import Path

from models import (PopularityRecommender, MatrixFactorizationRecommender,
                    RandomRecommender)
from evaluate import rmse, mae, evaluate_ranking
from audit_bias import audit_popularity_bias, compare_bias

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _require_raw_data():
    """
    data/raw/*.csv is gitignored (regenerable, not committed), so on a fresh
    clone these files are simply absent. Fail with the fix instead of a bare
    FileNotFoundError traceback.
    """
    missing = [name for name in ("ratings.csv", "movies.csv")
               if not (RAW_DIR / name).exists()]
    if missing:
        raise SystemExit(
            f"Missing input data: {', '.join(missing)} (expected in {RAW_DIR}).\n"
            "Generate the sample dataset first:\n"
            "    python src/generate_sample_data.py"
        )


def load_and_split(test_frac: float = 0.2):
    _require_raw_data()
    ratings = pd.read_csv(RAW_DIR / "ratings.csv")
    ratings = ratings.sort_values("timestamp").reset_index(drop=True)

    cutoff_idx = int(len(ratings) * (1 - test_frac))
    train_df = ratings.iloc[:cutoff_idx].copy()
    test_df = ratings.iloc[cutoff_idx:].copy()

    # Only evaluate on users/items seen during training -- otherwise we're
    # testing cold-start behavior, which is a different (real, but separate) problem
    train_users = set(train_df["user_id"])
    train_movies = set(train_df["movie_id"])
    test_df = test_df[test_df["user_id"].isin(train_users) & test_df["movie_id"].isin(train_movies)]

    return train_df, test_df


def build_seen_dict(train_df) -> dict:
    return train_df.groupby("user_id")["movie_id"].apply(set).to_dict()


def main():
    print("Loading and splitting data (chronological, leakage-free)...")
    train_df, test_df = load_and_split()
    print(f"  Train: {len(train_df)} ratings | Test: {len(test_df)} ratings")

    seen_by_user = build_seen_dict(train_df)
    all_movie_ids = sorted(pd.read_csv(RAW_DIR / "movies.csv")["movie_id"].unique())
    test_users = test_df["user_id"].unique().tolist()

    # --- Baselines ---
    print("\nTraining popularity baseline...")
    pop_model = PopularityRecommender(min_ratings=5).fit(train_df)
    rand_model = RandomRecommender(random_state=42).fit(train_df)

    # --- Matrix factorization (from scratch, SGD) ---
    print("Training matrix factorization model (SGD, this may take a moment)...")
    mf_model = MatrixFactorizationRecommender(
        n_factors=20, learning_rate=0.01, reg=0.02, n_epochs=20, random_state=42
    ).fit(train_df, verbose=True)

    # --- Rating-prediction accuracy (MF only -- popularity model doesn't predict individual ratings) ---
    print("\nEvaluating rating-prediction accuracy (Matrix Factorization)...")
    test_preds = [mf_model.predict(row.user_id, row.movie_id) for row in test_df.itertuples()]
    test_true = test_df["rating"].values

    # Reference point: predicting the training mean for every single rating.
    # An RMSE above this line means the model is worse than a constant, which is
    # exactly what a headline metric reported on its own will hide.
    baseline_preds = [mf_model.global_bias_] * len(test_df)
    baseline_rmse = rmse(test_true, baseline_preds)
    model_rmse = rmse(test_true, test_preds)

    print(f"  RMSE: {model_rmse:.4f}")
    print(f"  MAE:  {mae(test_true, test_preds):.4f}")
    print(f"  (reference) always-predict-global-mean RMSE: {baseline_rmse:.4f}")
    if model_rmse >= baseline_rmse:
        print("  WARNING: the model does not beat a constant predictor on RMSE.")
    else:
        improvement = (1 - model_rmse / baseline_rmse) * 100
        print(f"  -> {improvement:.1f}% better than the constant predictor.")

    # --- Ranking quality ---
    print("\nEvaluating ranking quality (Precision@10 / Recall@10)...")
    rand_ranking = evaluate_ranking(rand_model, test_df, seen_by_user, k=10)
    pop_ranking = evaluate_ranking(pop_model, test_df, seen_by_user, k=10)
    mf_ranking = evaluate_ranking(mf_model, test_df, seen_by_user, k=10)
    print(f"  Random (reference)   -- Precision@10: {rand_ranking['precision_at_k']:.4f}, "
          f"Recall@10: {rand_ranking['recall_at_k']:.4f} (n={rand_ranking['n_users_evaluated']} users)")
    print(f"  Popularity baseline  -- Precision@10: {pop_ranking['precision_at_k']:.4f}, "
          f"Recall@10: {pop_ranking['recall_at_k']:.4f} (n={pop_ranking['n_users_evaluated']} users)")
    print(f"  Matrix Factorization -- Precision@10: {mf_ranking['precision_at_k']:.4f}, "
          f"Recall@10: {mf_ranking['recall_at_k']:.4f} (n={mf_ranking['n_users_evaluated']} users)")
    if mf_ranking["precision_at_k"] <= rand_ranking["precision_at_k"]:
        print("  WARNING: MF does not beat random selection on Precision@10.")

    # --- Popularity-bias audit ---
    print("\nRunning popularity-bias audit...")
    pop_bias = audit_popularity_bias(pop_model, test_users, all_movie_ids, seen_by_user, k=10)
    mf_bias = audit_popularity_bias(mf_model, test_users, all_movie_ids, seen_by_user, k=10)
    rand_bias = audit_popularity_bias(rand_model, test_users, all_movie_ids, seen_by_user, k=10)
    print("\n" + compare_bias(pop_bias, mf_bias, rand_bias))

    # --- Persist the MF model ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "mf_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(mf_model, f)
    print(f"\nSaved trained model to {model_path}")


if __name__ == "__main__":
    main()
