"""
fetch_movielens.py

Downloads the real MovieLens 100K dataset and converts it into the schema the
rest of this project uses (the same schema generate_sample_data.py produces), so
every downstream script works unchanged.

    python src/fetch_movielens.py

100,000 ratings, 943 users, 1,682 movies. ~5MB, no signup required.

The raw files are not in a usable format for us:

  u.data  tab-separated:  user_id  item_id  rating  timestamp
  u.item  pipe-separated: movie_id|title|release_date|video_release|IMDb_URL|<19 genre flags>

Two details that break naive parsing:

1. `u.item` is **Latin-1**, not UTF-8. Several titles carry accented characters
   (e.g. "Les Misérables"), and reading them as UTF-8 raises UnicodeDecodeError.
2. Genres are 19 binary flag columns, not a genre name, and a film can carry
   several. We keep one genre per movie to match the existing schema -- see
   `_primary_genre` for what that discards.

Citation:
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent
Systems (TiiS) 5, 4, Article 19 (December 2015).
https://doi.org/10.1145/2827872
"""

import csv
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Column order of the 19 genre flags in u.item, taken from the dataset's u.genre
GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# u.item begins the genre flags at this field index
_FIRST_GENRE_FIELD = 5


def download(url: str = URL) -> zipfile.ZipFile:
    print(f"Downloading {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            payload = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            f"Could not download the dataset: {exc}\n"
            "Check your connection, or download the zip manually from\n"
            f"  {url}\n"
            "and extract u.data and u.item into data/raw/."
        )
    print(f"  received {len(payload):,} bytes")
    return zipfile.ZipFile(io.BytesIO(payload))


def _primary_genre(flags: list) -> str:
    """
    Collapse the 19 genre flags to a single label.

    MovieLens marks every genre that applies, so Toy Story is Animation AND
    Children's AND Comedy. Our schema carries one genre per movie, so we keep the
    first flagged genre in the dataset's own column order. This is a real
    simplification: it is display metadata only -- no model in this project reads
    the genre column, so it cannot affect any result.
    """
    for name, flag in zip(GENRE_COLUMNS, flags):
        if flag == "1" and name != "unknown":
            return name
    return "unknown"


def _release_year(release_date: str, title: str) -> str:
    """
    Year from the release_date field ("01-Jan-1995"), falling back to the year
    embedded in the title ("Toy Story (1995)"). A handful of rows have neither.
    """
    if release_date and "-" in release_date:
        return release_date.rsplit("-", 1)[-1]
    if title.endswith(")") and "(" in title:
        candidate = title[title.rfind("(") + 1:-1]
        if candidate.isdigit():
            return candidate
    return ""


def convert_movies(archive: zipfile.ZipFile, out_path: Path) -> int:
    # Latin-1, not UTF-8 -- accented titles raise UnicodeDecodeError otherwise.
    text = archive.read("ml-100k/u.item").decode("latin-1")

    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        movie_id, title, release_date = fields[0], fields[1], fields[2]
        genre = _primary_genre(fields[_FIRST_GENRE_FIELD:])
        rows.append([movie_id, title, genre, _release_year(release_date, title)])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["movie_id", "title", "genre", "release_year"])
        writer.writerows(rows)
    return len(rows)


def convert_ratings(archive: zipfile.ZipFile, out_path: Path) -> int:
    text = archive.read("ml-100k/u.data").decode("latin-1")

    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        user_id, movie_id, rating, timestamp = line.split("\t")
        rows.append([user_id, movie_id, rating, timestamp])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "movie_id", "rating", "timestamp"])
        writer.writerows(rows)
    return len(rows)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = download()

    n_movies = convert_movies(archive, RAW_DIR / "movies.csv")
    n_ratings = convert_ratings(archive, RAW_DIR / "ratings.csv")

    print(f"\nWrote REAL MovieLens 100K data to {RAW_DIR}")
    print(f"  movies.csv:  {n_movies:,} movies")
    print(f"  ratings.csv: {n_ratings:,} ratings")
    print("\nThis replaces the synthetic sample. Re-run training:")
    print("    python src/train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
