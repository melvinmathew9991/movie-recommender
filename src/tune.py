"""
tune.py

Hyperparameter search for the matrix factorization model, tracked in MLflow.

    python src/tune.py

WHY THIS EXISTS
---------------
`n_factors=20, learning_rate=0.01, reg=0.02` were chosen by hand and never
justified. This selects them on evidence instead.

WHY MLFLOW IS HERE
------------------
Not for decoration. A grid over three hyperparameters is a few dozen runs, and
comparing them by reading stdout does not work. MLflow is what makes the
comparison tractable, which is the only reason to add a tool.

It is a **development** dependency only (requirements-dev.txt). `train.py` never
imports it, the API image does not carry it, and CI does not run this script --
a grid is minutes of compute and CI should stay fast.

METHODOLOGY
-----------
- Every configuration is scored on the **validation** set.
- The epoch count is not part of the grid. Early stopping on validation finds it
  in a single run per configuration instead of one run per candidate value.
- The **test set is never touched here.** It is evaluated exactly once, by
  train.py, using the configuration this script selects. Selecting on test and
  then reporting test would make the reported number meaningless -- the same
  class of error this project already documents.
"""

import itertools
import json
import time
from pathlib import Path

import pandas as pd

from models import MatrixFactorizationRecommender
from split import per_user_chronological_split, split_summary

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
BEST_PARAMS_PATH = Path(__file__).resolve().parents[1] / "models" / "best_params.json"

MAX_EPOCHS = 40
PATIENCE = 3
RANDOM_STATE = 42

GRID = {
    "n_factors": [10, 20, 40],
    "learning_rate": [0.005, 0.01],
    "reg": [0.05, 0.1],
}


def _load_ratings():
    path = RAW_DIR / "ratings.csv"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}.\n"
            "Fetch the real dataset first:\n"
            "    python src/fetch_movielens.py"
        )
    return pd.read_csv(path)


def main():
    try:
        import mlflow
    except ImportError:
        raise SystemExit(
            "mlflow is not installed. It is a development dependency:\n"
            "    pip install -r requirements-dev.txt"
        )

    ratings = _load_ratings()
    train_df, val_df, test_df = per_user_chronological_split(ratings)
    print("Per-user chronological split:")
    print(split_summary(ratings, train_df, val_df, test_df))

    combos = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    print(f"\nSearching {len(combos)} configurations "
          f"(max {MAX_EPOCHS} epochs, early stopping patience {PATIENCE})")
    print("The test set is not touched by this script.\n")

    mlflow.set_experiment("movie-recommender-mf")
    results = []

    for i, params in enumerate(combos, start=1):
        started = time.time()
        with mlflow.start_run(run_name=f"mf-{i:02d}"):
            model = MatrixFactorizationRecommender(
                n_epochs=MAX_EPOCHS, random_state=RANDOM_STATE, **params
            ).fit(train_df, validation_df=val_df, patience=PATIENCE)

            elapsed = time.time() - started
            mlflow.log_params({**params, "max_epochs": MAX_EPOCHS,
                               "patience": PATIENCE, "random_state": RANDOM_STATE})
            mlflow.log_metrics({
                "val_rmse": model.best_val_rmse_,
                "best_epoch": model.best_epoch_,
                "seconds": round(elapsed, 1),
            })

            results.append({**params,
                            "best_epoch": model.best_epoch_,
                            "val_rmse": model.best_val_rmse_})
            print(f"  [{i:>2}/{len(combos)}] n_factors={params['n_factors']:<3} "
                  f"lr={params['learning_rate']:<6} reg={params['reg']:<5} "
                  f"-> val RMSE {model.best_val_rmse_:.4f} "
                  f"@ epoch {model.best_epoch_:<2} ({elapsed:.0f}s)")

    results.sort(key=lambda r: r["val_rmse"])
    best = results[0]

    print("\nTop 5 configurations by validation RMSE:")
    print(f"  {'n_factors':<10} {'lr':<8} {'reg':<7} {'epochs':<8} val_rmse")
    for r in results[:5]:
        print(f"  {r['n_factors']:<10} {r['learning_rate']:<8} {r['reg']:<7} "
              f"{r['best_epoch']:<8} {r['val_rmse']:.4f}")

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    print(f"\nBest configuration written to {BEST_PARAMS_PATH}")
    print("Run `python src/train.py` to train it and evaluate on the test set.")
    print("Browse the runs with `mlflow ui`.")


if __name__ == "__main__":
    main()
