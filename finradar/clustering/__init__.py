"""News clustering (cosine-similarity connected components)."""

from finradar.clustering.clusterer import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_KNN_CANDIDATES,
    DEFAULT_WINDOW_DAYS,
    cluster_recent_articles,
)

__all__ = [
    "DEFAULT_COSINE_THRESHOLD",
    "DEFAULT_KNN_CANDIDATES",
    "DEFAULT_WINDOW_DAYS",
    "cluster_recent_articles",
]
