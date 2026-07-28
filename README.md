# Movie Recommender System

A collaborative-filtering movie recommender built end to end: from-scratch
matrix factorization, a FastAPI serving layer, a Streamlit frontend, automated
testing and CI.

🚧 **Work in progress.** This README grows as each component lands.

## Quick start

```bash
pip install -r requirements-dev.txt   # runtime + test dependencies

python src/generate_sample_data.py    # writes data/raw/*.csv
pytest
```

## Data

The project builds against a **schema-accurate synthetic sample** matching the
MovieLens 100K format — 200 users, 300 movies, 15,000 ratings — so the pipeline
can be developed and tested without depending on an external download.

`data/raw/*.csv` is gitignored: it is regenerable output, not source. Run the
generator after cloning.

**This is not the real dataset.** Before publishing any results, swap in the
real MovieLens 100K data:
https://files.grouplens.org/datasets/movielens/ml-100k.zip

Movie popularity follows a Zipf-like distribution — a few titles get rated
heavily, most rarely — because a flat distribution would make the planned
popularity-bias audit meaningless.

## Models

**`PopularityRecommender`** — the naive baseline. Ranks by global mean rating
among items with enough ratings to be meaningful, with zero personalization.

**`MatrixFactorizationRecommender`** — biased matrix factorization trained by
SGD, written from scratch in NumPy rather than calling `scikit-surprise` or
`implicit`:

```
predicted_rating(u, i) = global_bias + user_bias[u] + item_bias[i]
                          + dot(user_factors[u], item_factors[i])
```

The bias terms matter: some users rate generously, some films are simply better,
and the latent factors shouldn't have to spend capacity representing that.
Unseen users or items fall back to the global mean rather than crashing.

**Why a baseline at all?** Without one, a recommender project can't demonstrate
that personalization is happening — a model can score well on aggregate metrics
while just reproducing what's already popular. The baseline exists to make that
comparison possible.

## Evaluation

Two metric families, because they measure different things:

- **Rating prediction** (RMSE, MAE) — how close the predicted star rating was
- **Ranking quality** (Precision@10, Recall@10) — of the ten items we'd actually
  show, how many did the user like? This is what matters operationally

The train/test split is **chronological, not random**. A random split would put
future ratings in the training set and leak information backwards.

## Popularity-bias audit

Headline metrics hide a well-known recommender failure mode: a model can score
well while just recommending whatever is already popular to everyone. The audit
measures **catalog coverage** (what fraction of the catalog ever gets
recommended) and **top-item share** (how concentrated recommendations are).

## Results (synthetic data)

**Rating prediction (Matrix Factorization):**
- Test RMSE: **1.1998** | Test MAE: **0.9718**
- Train RMSE fell to 0.8846 by epoch 20

**Ranking quality (n=200 users):**

| Model | Precision@10 | Recall@10 |
|---|---|---|
| Popularity baseline | 0.0125 | 0.0111 |
| Matrix Factorization | **0.0610** | **0.0630** |

MF outperforms the naive baseline by roughly 5× on both ranking metrics.

**Popularity-bias audit:**

| Metric | Popularity baseline | Matrix Factorization |
|---|---|---|
| Catalog coverage | 5.0% | **63.0%** |
| Top-item recommendation share | 9.6% | 4.6% |

**Finding:** MF's catalog coverage (63%) is dramatically broader than the
popularity baseline's (5%), and no single item dominates its recommendation
slots. Evidence of per-user personalization rather than the model reproducing
global popularity under a different name.

## Planned

- [x] Sample data generator
- [x] Popularity baseline + matrix factorization model
- [x] Evaluation metrics and popularity-bias audit
- [ ] FastAPI serving layer
- [ ] Streamlit frontend
- [ ] Containerization and CI

## License

MIT — see [LICENSE](LICENSE).
