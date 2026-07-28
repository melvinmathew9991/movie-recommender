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

## Planned

- [x] Sample data generator
- [x] Popularity baseline + matrix factorization model
- [ ] Evaluation metrics and popularity-bias audit
- [ ] FastAPI serving layer
- [ ] Streamlit frontend
- [ ] Containerization and CI

## License

MIT — see [LICENSE](LICENSE).
