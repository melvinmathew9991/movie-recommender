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

import random
import csv
from pathlib import Path

random.seed(42)

N_USERS = 200
N_MOVIES = 300
N_RATINGS = 15_000  # real ml-100k is 100K ratings / 943 users / 1682 movies

GENRES = ["Action", "Comedy", "Drama", "Romance", "Sci-Fi", "Thriller",
          "Horror", "Animation", "Documentary", "Crime"]


def generate_movies(out_path: Path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["movie_id", "title", "genre", "release_year"])
        for movie_id in range(1, N_MOVIES + 1):
            genre = random.choice(GENRES)
            year = random.randint(1990, 2020)
            writer.writerow([movie_id, f"Movie_{movie_id:04d}", genre, year])


def generate_ratings(out_path: Path):
    """
    Simulates realistic rating behavior: a power-law popularity distribution
    (a few movies get rated a lot, most get rated rarely) -- this matters
    later for the popularity-bias audit, since a flat/uniform distribution
    would make that audit meaningless.
    """
    # Zipf-like popularity weights: movie i's popularity ~ 1/i
    movie_weights = [1.0 / i for i in range(1, N_MOVIES + 1)]
    movie_ids = list(range(1, N_MOVIES + 1))

    seen_pairs = set()
    rows = []
    while len(rows) < N_RATINGS:
        user_id = random.randint(1, N_USERS)
        movie_id = random.choices(movie_ids, weights=movie_weights, k=1)[0]
        if (user_id, movie_id) in seen_pairs:
            continue
        seen_pairs.add((user_id, movie_id))

        # Ratings skew slightly positive (realistic: people rate what they choose to watch)
        rating = random.choices([1, 2, 3, 4, 5], weights=[5, 10, 20, 35, 30])[0]
        timestamp = random.randint(978300760, 1104537600)  # ~2001-2004 range, like real ml-100k
        rows.append([user_id, movie_id, rating, timestamp])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "movie_id", "rating", "timestamp"])
        writer.writerows(rows)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1] / "data" / "raw"
    base.mkdir(parents=True, exist_ok=True)

    generate_movies(base / "movies.csv")
    generate_ratings(base / "ratings.csv")

    print(f"Wrote SAMPLE data to {base}")
    print(f"  movies.csv:  {N_MOVIES} movies")
    print(f"  ratings.csv: {N_RATINGS} ratings from {N_USERS} users")
    print("Replace with the real MovieLens 100K dataset before using results externally.")
    print("Download: https://files.grouplens.org/datasets/movielens/ml-100k.zip")
