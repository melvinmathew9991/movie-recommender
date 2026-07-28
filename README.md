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

## ⚠️ The reported results do not hold up

Every metric this project published was reported without a reference point. Two
were added — predicting the global mean for every rating, and recommending at
random — and the previous conclusions collapse:

**Rating prediction:**

| Metric | Value |
|---|---|
| MF test RMSE | 1.1998 |
| Always-predict-global-mean RMSE | **1.1246** |

**The model is worse than a constant.** Predicting the same number for every
single rating beats it.

**Ranking quality (n=200 users):**

| Model | Precision@10 | Recall@10 |
|---|---|---|
| Random (reference) | 0.0370 | 0.0396 |
| Popularity baseline | 0.0125 | 0.0111 |
| Matrix Factorization | 0.0610 | 0.0630 |

The previously advertised "5× better than the baseline" was measured against a
baseline that **scores worse than random selection**. Against random — the
correct floor — MF's margin is 1.6×, not 5×.

**Popularity-bias audit:**

| Metric | Popularity baseline | Matrix Factorization | Random (reference) |
|---|---|---|---|
| Catalog coverage | 5.0% | 63.0% | **99.7%** |
| Top-item recommendation share | 9.6% | 4.6% | 0.9% |

Catalog coverage of 63% was presented as evidence of personalization. Random
selection reaches 99.7%. Breadth alone was never evidence of anything.

**What this means:** the numbers were never wrong arithmetically — they were
uninterpretable. A metric with no reference point cannot distinguish a model
that learned something from one that learned nothing. The next step is finding
out *why* the model can't beat a constant.

The reference baselines are now permanent parts of the pipeline, and training
prints an explicit `WARNING` if the model ever fails to beat either.

## Planned

- [x] Sample data generator
- [x] Popularity baseline + matrix factorization model
- [x] Evaluation metrics and popularity-bias audit
- [ ] FastAPI serving layer
- [ ] Streamlit frontend
- [ ] Containerization and CI

## License

MIT — see [LICENSE](LICENSE).
