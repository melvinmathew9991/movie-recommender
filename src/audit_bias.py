"""
audit_bias.py

Don't just report the headline accuracy metric -- go looking for what it hides.

The specific, well-known failure mode for recommender systems: a model
can achieve good RMSE/Precision@K while simply recommending whatever's
already popular to everyone -- providing zero real personalization.
This audit measures that directly, rather than assuming personalization
happened just because the model isn't the naive popularity baseline.

Metric used: catalog coverage (what fraction of the full catalog ever
gets recommended to anyone) and recommendation concentration (what
fraction of all recommendation slots go to the single most-recommended
item). A model with severe popularity bias will recommend a tiny subset
of the catalog to nearly everyone, regardless of individual user taste.
"""

from collections import Counter


def audit_popularity_bias(model, users: list, all_movie_ids: list,
                           train_seen_by_user: dict, k: int = 10) -> dict:
    all_recs = []
    for user_id in users:
        seen = train_seen_by_user.get(user_id, set())
        recs = model.recommend(user_id, k=k, exclude_seen=seen)
        all_recs.extend(recs)

    if not all_recs:
        return {"catalog_coverage": 0.0, "top_item_share": 0.0, "n_recommendations": 0}

    counts = Counter(all_recs)
    catalog_coverage = len(counts) / len(all_movie_ids)
    top_item_share = counts.most_common(1)[0][1] / len(all_recs)

    return {
        "catalog_coverage": float(catalog_coverage),
        "top_item_share": float(top_item_share),
        "n_unique_items_recommended": len(counts),
        "n_total_catalog_items": len(all_movie_ids),
        "n_recommendations": len(all_recs),
        "most_recommended_item": counts.most_common(1)[0][0],
    }


# Column widths. The label column must fit the longest metric name
# ("Top-item recommendation share", 29 chars) -- at 28 it overflowed and shifted
# that row out of alignment with the header.
_LABEL_W = 30
_COL_W = 22

# A model must beat the naive baseline's coverage by more than this factor before
# its breadth counts as evidence of anything.
_COVERAGE_MARGIN = 1.2


def _row(label: str, *values: str) -> str:
    cells = " ".join(f"{v:<{_COL_W}}" for v in values)
    return f"{label:<{_LABEL_W}} {cells}".rstrip()


def compare_bias(popularity_results: dict, mf_results: dict,
                 random_results: dict = None) -> str:
    """
    Human-readable comparison of the models' bias profiles.

    `random_results` is optional but strongly recommended: catalog coverage has
    no meaningful scale without it. Random selection covers nearly the whole
    catalog, so it is the ceiling against which MF's coverage should be read --
    high coverage rules out popularity collapse, it does not by itself prove
    personalization.
    """
    def cols(pop_val, mf_val, rnd_val):
        parts = [pop_val, mf_val]
        if random_results is not None:
            parts.append(rnd_val)
        return parts

    lines = [
        "Popularity-bias audit: comparing catalog coverage between models.",
        "",
        _row("Metric", *cols("Popularity baseline", "Matrix Factorization",
                             "Random (reference)")),
        _row("Catalog coverage",
             *cols(f"{popularity_results['catalog_coverage']:.1%}",
                   f"{mf_results['catalog_coverage']:.1%}",
                   f"{random_results['catalog_coverage']:.1%}" if random_results else "")),
        _row("Top-item recommendation share",
             *cols(f"{popularity_results['top_item_share']:.1%}",
                   f"{mf_results['top_item_share']:.1%}",
                   f"{random_results['top_item_share']:.1%}" if random_results else "")),
    ]
    if mf_results["catalog_coverage"] <= popularity_results["catalog_coverage"] * _COVERAGE_MARGIN:
        lines.append("")
        lines.append("FINDING: MF's catalog coverage is not meaningfully better than the naive "
                      "popularity baseline. The model may be learning to approximate global "
                      "popularity rather than genuine per-user personalization -- worth "
                      "investigating before claiming personalization as a feature.")
    else:
        lines.append("")
        lines.append("FINDING: MF's catalog coverage is meaningfully broader than the popularity "
                      "baseline, so it is not simply reproducing global popularity under another "
                      "name -- the failure mode this audit exists to catch.")
        if random_results is not None:
            lines.append("")
            lines.append("Read this against the random reference, not in isolation: random "
                          f"selection reaches {random_results['catalog_coverage']:.1%} coverage, so "
                          "breadth alone is not a personalization result. The evidence that the "
                          "targeting is any GOOD is Precision@K measured against that same "
                          "random reference.")
    return "\n".join(lines)
