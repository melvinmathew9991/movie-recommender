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


def compare_bias(popularity_results: dict, mf_results: dict) -> str:
    """Human-readable comparison of the two models' bias profiles."""
    lines = [
        "Popularity-bias audit: comparing catalog coverage between models.",
        "",
        f"{'Metric':<28} {'Popularity baseline':<22} {'Matrix Factorization':<22}",
        f"{'Catalog coverage':<28} {popularity_results['catalog_coverage']:.1%}"
        f"{'':<16} {mf_results['catalog_coverage']:.1%}",
        f"{'Top-item recommendation share':<28} {popularity_results['top_item_share']:.1%}"
        f"{'':<16} {mf_results['top_item_share']:.1%}",
    ]
    if mf_results["catalog_coverage"] <= popularity_results["catalog_coverage"] * 1.2:
        lines.append("")
        lines.append("FINDING: MF's catalog coverage is not meaningfully better than the naive "
                      "popularity baseline. The model may be learning to approximate global "
                      "popularity rather than genuine per-user personalization -- worth "
                      "investigating before claiming personalization as a feature.")
    else:
        lines.append("")
        lines.append("FINDING: MF shows meaningfully broader catalog coverage than the popularity "
                      "baseline, consistent with genuine per-user personalization rather than "
                      "just reproducing global popularity.")
    return "\n".join(lines)
