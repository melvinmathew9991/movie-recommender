# Movie Recommender System

A collaborative-filtering movie recommender built end to end: from-scratch
matrix factorization, a FastAPI serving layer, a Streamlit frontend, automated
testing and CI.

🚧 **Work in progress.** This README grows as each component lands.

## Quick start

```bash
pip install -r requirements-dev.txt   # runtime + test dependencies

python src/fetch_movielens.py         # real MovieLens 100K (needs network)
python src/tune.py                    # optional: hyperparameter search (~10 min)
python src/train.py
pytest
```

`train.py` uses `models/best_params.json` when `tune.py` has produced it, and
falls back to documented defaults otherwise — it prints which.

## Data

The results below are on the **real MovieLens 100K dataset** — 100,000 ratings
from 943 users on 1,682 movies, collected by GroupLens Research at the
University of Minnesota.

`python src/fetch_movielens.py` downloads it and converts it into this project's
schema. The raw files need real conversion work: ratings are tab-separated,
movies are pipe-separated with 19 binary genre flags rather than a genre name,
the release year is buried in a `01-Jan-1995` date field, and **`u.item` is
Latin-1** — reading it as UTF-8 fails on titles like *Les Misérables*.

### A synthetic generator also ships

`python src/generate_sample_data.py` produces a schema-accurate synthetic sample
(200 users, 300 movies, 15,000 ratings) from a latent taste model. This is what
**CI uses**: tests stay fast and hermetic, with no network dependency and no
5 MB download on every run.

`data/raw/*.csv` is gitignored either way — it is regenerable output, not source.

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

## Results — real MovieLens 100K

100,000 ratings, 943 users, 1,682 movies. Per-user chronological split; the test
set is evaluated exactly once, with the configuration selected on validation.

**Split** (`src/split.py`) — each user's 10 most recent ratings held out for
test, the 10 before those for validation:

| | Ratings | Users |
|---|---|---|
| Train | 85,120 | 943 |
| Validation | 7,429 | 744 |
| Test | 7,422 | **744 of 943 (79%)** |

**Selected hyperparameters** (`src/tune.py`, 12 configurations, chosen on
validation RMSE):

| Parameter | Value | Searched |
|---|---|---|
| `n_factors` | **40** | 10, 20, 40 |
| `learning_rate` | **0.01** | 0.005, 0.01 |
| `reg` | **0.1** | 0.05, 0.1 |
| epochs | **38** | early stopping, cap 40 |

Best validation RMSE 0.9535. The winning configuration sat at the edge of the
grid on two axes, so the boundary was probed rather than assumed: `reg` 0.15 and
0.2 scored worse (0.9608, 0.9813), `n_factors=80` was no better (0.9545), and
re-running the winner with a 120-epoch cap still selected epoch 38. The optimum
is interior.

**Rating prediction (Matrix Factorization):**

| Metric | Value |
|---|---|
| Test RMSE | **0.9998** |
| Test MAE | **0.7923** |
| Always-predict-global-mean RMSE (reference) | 1.2006 |
| Improvement over constant predictor | **16.7%** |

**Ranking quality (n=700 users):**

| Model | Precision@10 | Recall@10 |
|---|---|---|
| Random (reference) | 0.0040 | 0.0067 |
| Popularity baseline | **0.0213** | 0.0305 |
| Matrix Factorization | 0.0211 | **0.0329** |

### ⚠️ Tuning improved RMSE. It did not make MF beat the popularity baseline.

**On a 700-user evaluation, matrix factorization and the naive popularity
baseline are indistinguishable on Precision@10** — 0.0211 against 0.0213. MF
wins narrowly on Recall@10 (0.0329 vs 0.0305) and both clearly beat random, but
the headline "MF beats the baseline" does not survive a properly sized
evaluation.

The earlier 1.3× precision advantage was measured over 95 users. It did not
hold at 700.

This is a **known and well-documented result**, not a bug: models trained to
minimise RMSE optimise rating accuracy, which is not the same objective as
ranking the top-10 items a user will actually engage with. Cremonesi et al.
(RecSys 2010) showed simple popularity beating latent-factor models on top-N
tasks for exactly this reason. Fixing it means changing the objective — pairwise
ranking loss such as BPR, trained on implicit feedback — not tuning the current
one harder.

**What tuning did buy:** RMSE improved from 16.7% better than a constant
predictor, with hyperparameters now selected on evidence rather than asserted.
That is a real gain on the objective the model is actually trained for.

> **Numbers here are not comparable to earlier versions of this README.** The
> split protocol changed (global-timestamp → per-user), so the test set, its
> size and its user population are all different. Comparing 0.9998 against the
> previous 0.9817 would be meaningless — different denominators.

**Popularity-bias audit:**

| Metric | Popularity baseline | Matrix Factorization | Random (reference) |
|---|---|---|---|
| Catalog coverage | 2.3% | **11.4%** | 97.1% |
| Top-item recommendation share | 9.9% | 9.6% | 0.2% |

**Finding:** MF's coverage sits between the popularity baseline and random —
what a targeting model should look like. Coverage alone proves only that MF
isn't reproducing popularity under another name; the evidence that the targeting
is any *good* is Precision@10, which as shown above is a tie.

### Resolved: the global split discarded 86% of the test set

The previous global-timestamp split cut the dataset at the 80th percentile of
time, then dropped test rows whose user or movie was unseen in training. On real
MovieLens that left **95 of 943 users** — real users *arrive over time*, so the
final 20% of ratings is dominated by accounts that did not exist during the
training window. The synthetic data hid this completely: all 200 users there were
active across the whole time range.

The per-user split fixes it — **744 users evaluated instead of 95** — and it is
what exposed the ranking result above.

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
| Test suite | ✅ **57 tests passing** |
| Hyperparameter search (`tune.py`) | ✅ Run — 12 configurations tracked in MLflow, boundary probed |
| Real MovieLens converter | ✅ Run — 100,000 ratings / 943 users / 1,682 movies, parsing covered by hermetic tests |
| FastAPI service | ✅ Run under uvicorn — all three endpoints serve correctly |
| Streamlit frontend | ✅ Script executed and asserted via Streamlit's `AppTest` |
| Dockerfile | ✅ **Built and smoke-tested in CI** — image assembles, container starts, `/health` reports `model_loaded: true`, `/recommend` serves. Not built on the dev machine (no Docker daemon there) |
| docker-compose | ⚠️ Written, **never run** — Compose is not exercised by CI |
| GitHub Actions CI | ✅ Runs on every pull request |

The distinction between the Dockerfile and Compose is deliberate. CI builds the
image and asserts it actually serves, so the container layer is genuinely
verified — but it verifies the *single API image*, not the two-service Compose
topology. Nothing has run `docker compose up`, so that row stays ⚠️.

Building an image proves it assembles; running it and calling its endpoints
proves it works. The CI job does both, which is why this row moved from
"written" to "verified".

## Architecture

```
movie-recommender/
├── data/raw/                   # source CSVs (gitignored — regenerate)
├── src/
│   ├── fetch_movielens.py      # downloads + converts the real MovieLens 100K
│   ├── generate_sample_data.py # latent-factor synthetic data generator (CI)
│   ├── models.py               # Popularity + Random baselines, from-scratch MF
│   ├── evaluate.py             # RMSE, MAE, Precision@K, Recall@K
│   ├── audit_bias.py           # popularity-bias / filter-bubble audit
│   ├── split.py                # per-user chronological train/val/test split
│   ├── tune.py                 # hyperparameter search, tracked in MLflow
│   └── train.py                # split → train → evaluate → audit → persist
├── api/main.py                 # FastAPI serving layer
├── app/streamlit_app.py        # frontend, calls the API
├── tests/                      # 57 tests: models, split, data, converter, frontend
├── models/                     # persisted artifacts (gitignored)
├── .github/workflows/ci.yml    # test + docker build + container smoke test
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml              # pytest configuration
```

## SDLC & MLOps practices applied

- **Version control** — every change landed through a reviewed pull request with
  a linked issue; `main` protected behind passing CI
- **Experiment tracking** — hyperparameters selected on a validation set with
  MLflow, never on test. A dev dependency only: `train.py` and the API never
  import it, and CI does not run the search
- **Automated testing** — 57 tests covering metric correctness, model behaviour
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
- **MF does not beat the popularity baseline on ranking.** See Results. The model
  is trained on an RMSE objective, which is not the ranking objective. Closing
  this needs a pairwise ranking loss (BPR) on implicit feedback, not more tuning.
- **199 of 943 users are excluded from evaluation** — those with fewer than 30
  ratings, who cannot give up 20 held-out items and still be learnable. Their
  ratings still train the item factors; they are simply never scored.
- **The split is leakage-free per user, not globally time-ordered.** One user's
  training ratings may post-date another's test ratings. This is standard
  practice in the recommender literature (MovieLens ships `ua.base`/`ua.test`
  built the same way); strict global ordering costs 89% of the evaluation set.
- **`recommend()` scores every candidate in a Python loop**, which would not
  scale. Precomputing top-K offline or using an approximate nearest-neighbour
  index over the item factors is the standard fix.

## Next steps

1. **Train on a ranking objective (BPR on implicit feedback).** The evaluation
   above shows RMSE-optimised MF tying the popularity baseline on Precision@10.
   That is an objective mismatch, and it is the single most valuable thing left
   to fix.
2. Exercise the Compose topology — CI verifies the API image, but not the
   two-service stack
3. Fall back to the popularity baseline for cold-start users
4. Add item-based collaborative filtering as a fourth comparison point

## License

MIT — see [LICENSE](LICENSE).
