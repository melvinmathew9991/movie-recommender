"""
main.py -- FastAPI serving layer for the trained recommender.

Consistent with the FastAPI deployment pattern already used in the
Nectar and AML projects: a thin, typed API wrapping a loaded model
artifact, with a health check and structured logging for monitoring.

Run locally with `uvicorn api.main:app --reload` from the project root.
Verified running: /health, /recommend/{user_id} and /predict all serve
correctly against a model trained by `python src/train.py`.
"""

import sys
import pickle
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Upper bound on top-K. Guards against a single request scoring and sorting the
# entire catalog; the sample catalog is small, but this endpoint is the one that
# would face real traffic.
MAX_K = 100

# The pickled model references the `models` module (src/models.py) by name, so
# that module must be importable before pickle.load runs. Two things break
# without this line: src/ isn't on the path at all when uvicorn is launched
# from the project root, and the models/ ARTIFACT DIRECTORY sitting there would
# otherwise be the only `models` the import system can see -- it resolves as an
# empty namespace package, and unpickling dies with a confusing AttributeError
# rather than a missing-module error. A real module beats a namespace package
# regardless of path order, so inserting src/ resolves both.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommender-api")

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mf_model.pkl"

_model = None  # loaded on startup, held in module state


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    logger.info(f"Loading model from {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Movie Recommender API",
    description="Serves top-K movie recommendations from a from-scratch matrix factorization model.",
    version="1.0.0",
    lifespan=lifespan,
)


class RecommendationResponse(BaseModel):
    user_id: int
    recommended_movie_ids: list[int]
    model_version: str = "mf-v1"


class PredictionResponse(BaseModel):
    user_id: int
    movie_id: int
    predicted_rating: float


@app.get("/health")
def health_check():
    """Basic liveness + model-loaded check for monitoring/orchestration."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: int, k: int = Query(10, ge=1, le=MAX_K)):
    """
    Top-K recommendations for a user.

    Cold start: a user_id absent from training has no learned bias or factors,
    so every candidate scores at the global mean and the returned ordering
    carries no personalization signal. The endpoint still answers rather than
    erroring -- see the README for why this is surfaced rather than hidden.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    seen = set()  # in production this would query a real "seen items" store
    try:
        recs = _model.recommend(user_id, k=k, exclude_seen=seen)
    except Exception as e:
        logger.error(f"Recommendation failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Recommendation generation failed")

    logger.info(f"Served {len(recs)} recommendations to user {user_id}")
    return RecommendationResponse(user_id=user_id, recommended_movie_ids=recs)


@app.get("/predict", response_model=PredictionResponse)
def predict_rating(user_id: int, movie_id: int):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pred = _model.predict(user_id, movie_id)
    return PredictionResponse(user_id=user_id, movie_id=movie_id, predicted_rating=pred)
