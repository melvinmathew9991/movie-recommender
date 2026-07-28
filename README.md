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

## The defect that invalidated every earlier result

The reference baselines showed the model was worse than a constant predictor.
The cause turned out not to be the model.

`generate_sample_data.py` drew every rating from a single fixed distribution,
independent of **both** the user and the movie. The dataset contained no
user–item preference structure at all, so there was nothing for a recommender to
recover.

Confirmed by permutation test — shuffling the ratings at random and recomputing
the per-user and per-item spread produced *the same or greater* spread than the
real data:

| Statistic | Before fix | After fix |
|---|---|---|
| Per-user mean-rating spread | **z = −2.59** | z = +27.92 |
| Per-item mean-rating spread | **z = −1.75** | z = +29.72 |

A negative z means the observed structure is indistinguishable from noise.

**Fixed** by generating ratings from a latent taste model — user preference
vectors, movie attribute vectors, bias terms, and a popularity/quality
correlation — deliberately the same family matrix factorization tries to
recover, so the pipeline is exercised against data where personalization
genuinely exists.

### What the numbers were, and what they became

| Measure | Before | After |
|---|---|---|
| MF test RMSE vs constant predictor | 1.1998 vs 1.1246 — **worse than a constant** | 0.8288 vs 1.0005 — **17.2% better** |
| MF Precision@10 vs random | 0.0610 vs 0.0370 | 0.1495 vs 0.0355 |
| Popularity baseline Precision@10 | 0.0125 — **below random** | 0.0960 — a real competitor |

The metrics were never arithmetically wrong. They were uninterpretable, and the
data underneath them was empty.

<details>
<summary>The original, invalidated results (kept for the record)</summary>

## ⚠️ The reported results did not hold up

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

</details>

## Results (synthetic data — re-validate on real MovieLens 100K)

**Rating prediction (Matrix Factorization):**

| Metric | Value |
|---|---|
| Test RMSE | **0.8288** |
| Test MAE | **0.6682** |
| Always-predict-global-mean RMSE (reference) | 1.0005 |
| Improvement over constant predictor | **17.2%** |

**Ranking quality (n=200 users):**

| Model | Precision@10 | Recall@10 |
|---|---|---|
| Random (reference) | 0.0355 | 0.0377 |
| Popularity baseline | 0.0960 | 0.1110 |
| Matrix Factorization | **0.1495** | **0.1802** |

MF beats random by 4.2× and the popularity baseline by 1.6×. The second
comparison is the meaningful one — beating random only proves the model learned
*something*.

**Popularity-bias audit:**

| Metric | Popularity baseline | Matrix Factorization | Random (reference) |
|---|---|---|---|
| Catalog coverage | 7.3% | **32.3%** | 97.7% |
| Top-item recommendation share | 9.2% | 7.0% | 0.8% |

**Finding:** MF's coverage sits between the popularity baseline and random,
which is what a targeting model should look like — concentrating on relevant
items without collapsing onto the globally popular ones. Coverage alone proves
only that MF isn't reproducing popularity under another name; the evidence that
the targeting is any *good* is Precision@10 against the same random reference.

These numbers reproduce bit-identically from a clean regenerate-and-retrain.

## Planned

- [x] Sample data generator
- [x] Popularity baseline + matrix factorization model
- [x] Evaluation metrics and popularity-bias audit
- [ ] FastAPI serving layer
- [ ] Streamlit frontend
- [ ] Containerization and CI

## License

MIT — see [LICENSE](LICENSE).
