"""News clustering (cosine-similarity connected components)."""

from finradar.clustering.clusterer import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_KNN_CANDIDATES,
    DEFAULT_MIN_BODY_JACCARD,
    DEFAULT_SAME_LANG_COSINE,
    DEFAULT_TITLE_OVERLAP_MIN,
    DEFAULT_TITLE_OVERLAP_RATIO,
    DEFAULT_WINDOW_DAYS,
    cluster_recent_articles,
)

__all__ = [
    "DEFAULT_COSINE_THRESHOLD",
    "DEFAULT_KNN_CANDIDATES",
    "DEFAULT_MIN_BODY_JACCARD",
    "DEFAULT_SAME_LANG_COSINE",
    "DEFAULT_TITLE_OVERLAP_MIN",
    "DEFAULT_TITLE_OVERLAP_RATIO",
    "DEFAULT_WINDOW_DAYS",
    "cluster_recent_articles",
]
