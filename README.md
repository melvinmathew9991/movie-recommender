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

## Serving

**`api/main.py`** — FastAPI service loading the persisted model at startup:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness plus a model-loaded check, for monitoring |
| `GET /recommend/{user_id}?k=10` | Top-K recommendations |
| `GET /predict?user_id=&movie_id=` | Predicted rating for one pair |

**`app/streamlit_app.py`** — frontend that calls the API rather than importing
the model directly, so the service boundary is real. It reads `API_BASE` from the
environment, which is what lets the same code work locally and under Compose.

## Containerization

`Dockerfile` containerizes the API; `docker-compose.yml` runs the API and
frontend as separate services. Both the model artifact and `data/raw/` are
gitignored, so **`python src/train.py` must run before `docker build`** — CI does
exactly that.

## What's verified vs. what isn't

| Component | Status |
|---|---|
| Data generation | ✅ Run and verified |
| Matrix factorization from scratch | ✅ Run and verified |
| Evaluation metrics | ✅ Run and verified |
| Popularity-bias audit | ✅ Run and verified |
| Training pipeline | ✅ Runs end to end from a clean state |
| Test suite | ✅ **34 tests passing** |
| FastAPI service | ✅ Run under uvicorn — all three endpoints serve correctly |
| Streamlit frontend | ✅ Script executed and asserted via Streamlit's `AppTest` |
| Dockerfile / docker-compose | ⚠️ Written, **not built locally** — no Docker daemon on the dev machine; the CI `docker-build` job is what exercises them |
| GitHub Actions CI | ✅ Runs on every pull request |

This distinction is deliberate. The container layer is written and internally
consistent, but it has never been built on the development machine.

## Architecture

```
movie-recommender/
├── data/raw/                   # source CSVs (gitignored — regenerate)
├── src/
│   ├── generate_sample_data.py # latent-factor synthetic data generator
│   ├── models.py               # Popularity + Random baselines, from-scratch MF
│   ├── evaluate.py             # RMSE, MAE, Precision@K, Recall@K
│   ├── audit_bias.py           # popularity-bias / filter-bubble audit
│   └── train.py                # split → train → evaluate → audit → persist
├── api/main.py                 # FastAPI serving layer
├── app/streamlit_app.py        # frontend, calls the API
├── tests/                      # 34 tests across models, data and frontend
├── models/                     # persisted artifacts (gitignored)
├── .github/workflows/ci.yml    # test + docker build + container smoke test
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml              # pytest configuration
```

## SDLC & MLOps practices applied

- **Version control** — every change landed through a reviewed pull request with
  a linked issue; `main` protected behind passing CI
- **Automated testing** — 34 tests covering metric correctness, model behaviour
  (cold start, seen-item exclusion, unfitted-model errors), reproducibility,
  data-generation integrity and the rendered frontend. Each regression test was
  verified to **fail against the original code** — a regression test that cannot
  fail is worthless
- **CI/CD** — every PR runs the suite, a training smoke test, a Docker build, and
  a container smoke test asserting the image actually serves
- **Reproducibility** — seeded data generation and training, both test-enforced;
  a clean regenerate-and-retrain reproduces every number above bit-identically
- **Monitoring readiness** — `/health` endpoint, structured logging per request,
  Docker healthcheck polling the same endpoint
- **Semantic versioning** — `v0.1.0` core pipeline, `v0.2.0` corrected
  evaluation, `v1.0.0` complete

## Known limitations

- **Cold start is unhandled.** A `user_id` absent from training has no learned
  factors, so every candidate scores at the global mean and the ordering carries
  no personalization signal. The API answers rather than erroring. Falling back
  to the popularity baseline is the obvious fix.
- **`/recommend` ignores previously-seen items** — it passes an empty
  `exclude_seen`; production would query a real "seen items" store.
- **Results are on synthetic data**, generated by a latent-factor model of the
  same family MF recovers. That makes the pipeline honest to evaluate, but it
  also makes the task easier than real human rating behaviour.
- **`recommend()` scores every candidate in a Python loop**, which would not
  scale. Precomputing top-K offline or using an approximate nearest-neighbour
  index over the item factors is the standard fix.

## Next steps

1. Swap in the real MovieLens 100K dataset and re-run; replace the Results section
2. Build and run the container locally once Docker is available
3. Fall back to the popularity baseline for cold-start users
4. Add item-based collaborative filtering as a fourth comparison point

## License

MIT — see [LICENSE](LICENSE).
