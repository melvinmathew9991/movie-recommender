"""
test_fetch_movielens.py

Tests the MovieLens converter against a miniature in-memory archive built to
mirror the real file quirks. No network access: the download is never exercised
here, only the parsing, so this runs in CI like everything else.

The fixtures deliberately include the two things that break naive parsing --
a Latin-1 accented title, and a movie flagged with several genres.
"""

import csv
import io
import zipfile

import pytest

import fetch_movielens as fm


# movie_id|title|release_date|video_release|IMDb_URL|<19 genre flags>
# Flags follow u.genre order: unknown, Action, Adventure, Animation, Children's,
# Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical,
# Mystery, Romance, Sci-Fi, Thriller, War, Western
_ITEM_LINES = [
    # Animation + Children's + Comedy -> primary genre is Animation
    "1|Toy Story (1995)|01-Jan-1995||http://x|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0",
    # Action only
    "2|GoldenEye (1995)|01-Jan-1995||http://x|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0",
    # Accented title -- Latin-1 encoded, would raise UnicodeDecodeError as UTF-8
    "3|Les Misérables (1995)|01-Jan-1995||http://x|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0",
    # No release_date -- year must fall back to the title
    "4|Mystery Film (1987)|||http://x|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|0",
    # Only the 'unknown' flag set
    "5|Unclassified||| http://x|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0",
]

_DATA_LINES = [
    "196\t242\t3\t881250949",
    "186\t302\t3\t891717742",
    "22\t377\t1\t878887116",
]


@pytest.fixture
def archive():
    """A zip shaped like ml-100k.zip, encoded Latin-1 as the real one is."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ml-100k/u.item", "\n".join(_ITEM_LINES).encode("latin-1"))
        zf.writestr("ml-100k/u.data", "\n".join(_DATA_LINES).encode("latin-1"))
    buf.seek(0)
    return zipfile.ZipFile(buf)


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- Ratings ------------------------------------------------------------------


def test_ratings_convert_to_the_project_schema(archive, tmp_path):
    out = tmp_path / "ratings.csv"
    n = fm.convert_ratings(archive, out)
    rows = _read(out)

    assert n == 3
    assert list(rows[0].keys()) == ["user_id", "movie_id", "rating", "timestamp"]
    assert rows[0] == {"user_id": "196", "movie_id": "242",
                       "rating": "3", "timestamp": "881250949"}


# --- Movies -------------------------------------------------------------------


def test_movies_convert_to_the_project_schema(archive, tmp_path):
    out = tmp_path / "movies.csv"
    n = fm.convert_movies(archive, out)
    rows = _read(out)

    assert n == 5
    assert list(rows[0].keys()) == ["movie_id", "title", "genre", "release_year"]
    assert rows[0]["title"] == "Toy Story (1995)"
    assert rows[0]["release_year"] == "1995"


def test_latin1_titles_survive_the_conversion(archive, tmp_path):
    """Reading u.item as UTF-8 raises UnicodeDecodeError on accented titles."""
    out = tmp_path / "movies.csv"
    fm.convert_movies(archive, out)
    titles = [r["title"] for r in _read(out)]
    assert "Les Misérables (1995)" in titles


def test_multi_genre_movies_collapse_to_the_first_flagged_genre(archive, tmp_path):
    """Toy Story is Animation + Children's + Comedy; our schema holds one."""
    out = tmp_path / "movies.csv"
    fm.convert_movies(archive, out)
    rows = {r["movie_id"]: r for r in _read(out)}
    assert rows["1"]["genre"] == "Animation"
    assert rows["2"]["genre"] == "Action"


def test_unknown_is_never_chosen_over_a_real_genre(archive, tmp_path):
    out = tmp_path / "movies.csv"
    fm.convert_movies(archive, out)
    rows = {r["movie_id"]: r for r in _read(out)}
    assert rows["3"]["genre"] == "Drama"
    assert rows["5"]["genre"] == "unknown", "only when nothing else is flagged"


# --- Year parsing -------------------------------------------------------------


def test_release_year_falls_back_to_the_title_when_the_date_is_missing():
    assert fm._release_year("", "Mystery Film (1987)") == "1987"


def test_release_year_prefers_the_date_field():
    assert fm._release_year("01-Jan-1995", "Toy Story (1995)") == "1995"


def test_release_year_is_empty_when_neither_source_has_one():
    assert fm._release_year("", "Untitled") == ""
