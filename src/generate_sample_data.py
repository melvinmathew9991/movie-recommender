"""
generate_sample_data.py

Produces a SCHEMA-ACCURATE SAMPLE of the MovieLens 100K dataset so the rest
of this project can be built, tested and run end to end without depending on
an external download.

*** THIS IS NOT THE REAL DATASET. ***
Replace data/raw/ratings.csv and data/raw/movies.csv with the real files
before publishing any results or putting numbers on your resume.

Real dataset (MovieLens 100K, 100,000 ratings from 943 users on 1,682
movies, collected by GroupLens Research at the University of Minnesota):
https://grouplens.org/datasets/movielens/100k/
Direct download: https://files.grouplens.org/datasets/movielens/ml-100k.zip
No signup required, single zip file.

Citation:
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent
Systems (TiiS) 5, 4, Article 19 (December 2015).
"""

import csv
from pathlib import Path

import numpy as np

SEED = 42

N_USERS = 200
N_MOVIES = 300
N_RATINGS = 15_000  # real ml-100k is 100K ratings / 943 users / 1682 movies

GENRES = ["Action", "Comedy", "Drama", "Romance", "Sci-Fi", "Thriller",
          "Horror", "Animation", "Documentary", "Crime"]

# --- Latent preference structure -------------------------------------------
#
# Ratings are drawn from a low-rank taste model rather than one fixed
# distribution. This is not decoration: the previous version sampled every
# rating independently of both the user and the movie, which left the dataset
# with no recoverable preference signal at all. Matrix factorization then had
# nothing to learn, scored worse than predicting the global mean, and barely
# beat a random recommender -- while the evaluation still reported those numbers
# as though they demonstrated personalization.
#
# The generative model below is deliberately the same family the recommender
# tries to recover (biases plus a low-rank user/item interaction), so the
# pipeline is exercised against data where personalization genuinely exists.
N_LATENT = 3
GLOBAL_MEAN = 3.6
TASTE_SCALE = 0.35          # weight on the user-taste x movie-attribute interaction
USER_BIAS_SD = 0.30         # some users rate generously, some harshly
QUALITY_SD = 0.35           # idiosyncratic per-movie quality
POPULARITY_QUALITY = 0.40   # popular movies rate slightly higher, as in real data
NOISE_SD = 0.70             # irreducible error -- keeps the task non-trivial


def _latent_factors(rng):
    """User taste vectors, movie attribute vectors, and the bias terms."""
    user_taste = rng.normal(0, 1, (N_USERS + 1, N_LATENT))
    movie_attrs = rng.normal(0, 1, (N_MOVIES + 1, N_LATENT))
    user_bias = rng.normal(0, USER_BIAS_SD, N_USERS + 1)

    # movie_id 1 is the most popular (see the Zipf weights below); tie a modest
    # quality gradient to that ranking so the popularity baseline is a genuine
    # competitor rather than a straw man.
    rank = np.arange(N_MOVIES + 1)
    pop_component = POPULARITY_QUALITY * (1 - 2 * (rank - 1) / max(N_MOVIES - 1, 1))
    movie_quality = pop_component + rng.normal(0, QUALITY_SD, N_MOVIES + 1)

    return user_taste, movie_attrs, user_bias, movie_quality


def _true_score(u, i, user_taste, movie_attrs, user_bias, movie_quality):
    """The rating a user would give a movie, before observation noise."""
    interaction = float(np.dot(user_taste[u], movie_attrs[i]))
    return (GLOBAL_MEAN + user_bias[u] + movie_quality[i]
            + TASTE_SCALE * interaction)


def generate_movies(out_path: Path, rng):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["movie_id", "title", "genre", "release_year"])
        for movie_id in range(1, N_MOVIES + 1):
            genre = GENRES[rng.integers(0, len(GENRES))]
            year = int(rng.integers(1990, 2021))
            writer.writerow([movie_id, f"Movie_{movie_id:04d}", genre, year])


def generate_ratings(out_path: Path, rng):
    """
    Simulates realistic rating behavior on two axes:

    1. WHICH movies get rated -- a power-law popularity distribution (a few
       movies get rated a lot, most get rated rarely). This matters for the
       popularity-bias audit, since a flat distribution would make it meaningless.

    2. WHAT rating is given -- drawn from the latent taste model above, so a
       user's rating depends on their own preferences and the movie's
       attributes. This is what makes the ratings learnable.
    """
    user_taste, movie_attrs, user_bias, movie_quality = _latent_factors(rng)

    # Zipf-like popularity weights: movie i's popularity ~ 1/i
    movie_ids = np.arange(1, N_MOVIES + 1)
    weights = 1.0 / movie_ids
    weights = weights / weights.sum()

    seen_pairs = set()
    rows = []
    while len(rows) < N_RATINGS:
        user_id = int(rng.integers(1, N_USERS + 1))
        movie_id = int(rng.choice(movie_ids, p=weights))
        if (user_id, movie_id) in seen_pairs:
            continue
        seen_pairs.add((user_id, movie_id))

        score = _true_score(user_id, movie_id, user_taste, movie_attrs,
                            user_bias, movie_quality)
        rating = int(np.clip(round(score + rng.normal(0, NOISE_SD)), 1, 5))

        timestamp = int(rng.integers(978300760, 1104537600))  # ~2001-2004, like real ml-100k
        rows.append([user_id, movie_id, rating, timestamp])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "movie_id", "rating", "timestamp"])
        writer.writerows(rows)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1] / "data" / "raw"
    base.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    generate_movies(base / "movies.csv", rng)
    generate_ratings(base / "ratings.csv", rng)

    print(f"Wrote SAMPLE data to {base}")
    print(f"  movies.csv:  {N_MOVIES} movies")
    print(f"  ratings.csv: {N_RATINGS} ratings from {N_USERS} users")
    print("Replace with the real MovieLens 100K dataset before using results externally.")
    print("Download: https://files.grouplens.org/datasets/movielens/ml-100k.zip")
