"""
test_data_generation.py

Guards the sample data generator: the files it writes must match the MovieLens
schema, stay inside valid ranges, and be reproducible from the same seed.
"""

import csv
import random

import generate_sample_data as gen


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_movies_file_schema_and_ranges(tmp_path):
    out = tmp_path / "movies.csv"
    gen.generate_movies(out)
    rows = _read(out)

    assert len(rows) == gen.N_MOVIES
    assert list(rows[0].keys()) == ["movie_id", "title", "genre", "release_year"]
    assert {int(r["movie_id"]) for r in rows} == set(range(1, gen.N_MOVIES + 1))
    assert all(r["genre"] in gen.GENRES for r in rows)
    assert all(1990 <= int(r["release_year"]) <= 2020 for r in rows)


def test_ratings_file_schema_and_ranges(tmp_path):
    out = tmp_path / "ratings.csv"
    gen.generate_ratings(out)
    rows = _read(out)

    assert len(rows) == gen.N_RATINGS
    assert list(rows[0].keys()) == ["user_id", "movie_id", "rating", "timestamp"]
    assert all(1 <= int(r["rating"]) <= 5 for r in rows), "ratings must stay on the 1-5 scale"
    assert all(1 <= int(r["user_id"]) <= gen.N_USERS for r in rows)
    assert all(1 <= int(r["movie_id"]) <= gen.N_MOVIES for r in rows)


def test_a_user_never_rates_the_same_movie_twice(tmp_path):
    out = tmp_path / "ratings.csv"
    gen.generate_ratings(out)
    rows = _read(out)

    pairs = {(r["user_id"], r["movie_id"]) for r in rows}
    assert len(pairs) == len(rows), "duplicate (user, movie) pairs would corrupt the train/test split"


def test_generation_is_reproducible_from_the_same_seed(tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"

    random.seed(42)
    gen.generate_ratings(first)
    random.seed(42)
    gen.generate_ratings(second)

    assert first.read_bytes() == second.read_bytes(), "same seed must reproduce the dataset"
