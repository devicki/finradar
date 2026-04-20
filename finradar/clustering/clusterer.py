"""
finradar.clustering.clusterer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Group recent news articles into clusters of "same story" via connected
components over cosine similarity, with a title-token overlap guard to
suppress template-based false positives.

Algorithm
---------
1. Pull recent articles (last N days) that have an embedding.
2. For each article, use pgvector KNN to find up to K nearest neighbours
   with cosine similarity ≥ threshold.
3. **Overlap filter** — for each candidate edge, require at least one
   title token in common when both articles share a language. Skipped
   for cross-lingual pairs (token overlap is meaningless across scripts).
4. Union-Find over every surviving (article, neighbour) pair → disjoint sets.
5. For each component:
   * size == 1 → singleton. Clear cluster_rep_id (becomes NULL), cluster_size = 1.
   * size >= 2 → pick representative = newest ``last_seen_at`` member.
                 Compute similarity of every member to the representative.
6. Bulk UPDATE ``cluster_rep_id``, ``cluster_size``, ``similarity_to_rep``.

Why the overlap filter?
    The multilingual embedding places all Korean news in a relatively
    narrow region of vector space. Template-heavy patterns like
    "X 기관명 직책" (인사동정) or "[포토] XYZ" (스포츠 포토 캡션) land
    above cosine 0.95 despite reporting different people / events.
    Requiring a shared content token prevents template similarity from
    chaining unrelated stories.

Cross-lingual clustering
    The ``paraphrase-multilingual-MiniLM-L12-v2`` model projects Korean
    and English text into the same space, so semantically matching
    stories cluster across languages (e.g. "Fed raises rates" +
    "연준 금리 인상"). For those cross-lingual pairs there are no shared
    tokens, so the overlap filter is intentionally bypassed and cosine
    alone decides.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session


logger = logging.getLogger("finradar.clustering")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_DAYS: int = 7
DEFAULT_COSINE_THRESHOLD: float = 0.80
DEFAULT_KNN_CANDIDATES: int = 30
DEFAULT_TITLE_OVERLAP_MIN: int = 1  # required content-token overlap for same-lang pairs

# Tokens shorter than this are treated as noise (particles, prepositions, digits).
_MIN_TOKEN_LEN: int = 2

# Word boundary for title tokenisation — whitespace + punctuation commonly
# found in KO/EN headlines (…, [], (), quotes, dashes, etc.).
_TITLE_TOKEN_SPLIT = re.compile(r"[\s,.…·!?'\"\[\]\(\)\-—–]+")


# ---------------------------------------------------------------------------
# Union-Find (path compression + union by size)
# ---------------------------------------------------------------------------


class _UnionFind:
    """Disjoint-set forest for integer ids.

    Uses path compression on find() and union-by-size so amortised ops are
    effectively O(α(N)).  Nodes are added lazily on first reference.
    """

    __slots__ = ("parent", "size")

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.size: dict[int, int] = {}

    def _add(self, x: int) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x: int) -> int:
        self._add(x)
        # Iterative path compression
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Attach smaller tree under larger
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self, ids: Iterable[int]) -> dict[int, list[int]]:
        """Return {root_id: [member_ids...]} for the given ids."""
        groups: dict[int, list[int]] = defaultdict(list)
        for i in ids:
            groups[self.find(i)].append(i)
        return groups


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ClusteringResult:
    articles_considered: int
    edges_found: int
    edges_after_overlap: int
    non_singleton_clusters: int
    largest_cluster_size: int
    updated_rows: int
    elapsed_sec: float


# ---------------------------------------------------------------------------
# Title-overlap helpers
# ---------------------------------------------------------------------------


def _title_tokens(title: str | None) -> set[str]:
    """Extract content tokens (length ≥ 2) from a headline.

    Lowercased so English "Fed" and "fed" match; Korean text unaffected.
    """
    if not title:
        return set()
    return {
        tok.lower()
        for tok in _TITLE_TOKEN_SPLIT.split(title)
        if len(tok) >= _MIN_TOKEN_LEN
    }


def _pair_passes_overlap(
    lang_a: str | None,
    lang_b: str | None,
    tokens_a: set[str],
    tokens_b: set[str],
    min_overlap: int,
) -> bool:
    """True when a candidate edge survives the overlap filter.

    * Different languages → skip the check (cross-lingual pairs by design
      have no shared tokens; cosine alone must decide).
    * Same language → require at least ``min_overlap`` shared content tokens.
    """
    if (lang_a or "") != (lang_b or ""):
        return True
    return len(tokens_a & tokens_b) >= min_overlap


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def cluster_recent_articles(
    session: Session,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    threshold: float = DEFAULT_COSINE_THRESHOLD,
    knn_candidates: int = DEFAULT_KNN_CANDIDATES,
    title_overlap_min: int = DEFAULT_TITLE_OVERLAP_MIN,
) -> ClusteringResult:
    """Cluster all articles inside the recency window.

    Intended to be called from a Celery task with a synchronous SQLAlchemy
    Session (see ``collection_tasks.SyncSessionLocal``).

    Args:
        session:            Sync SQLAlchemy session (bound to psycopg2 engine).
        window_days:        Articles with ``last_seen_at`` within this many
                            days participate. Older clusters stay as-is.
        threshold:          Minimum cosine similarity for two articles to
                            be considered "same story".
        knn_candidates:     Per-article neighbour cap. Higher → denser edges,
                            more CPU. 30 is plenty for news dedup.
        title_overlap_min:  Required number of shared title tokens for
                            same-language pairs. 1 suppresses template-based
                            false positives (인사동정, 포토 캡션 등). Set to
                            0 to disable the overlap filter entirely.

    Returns:
        Summary statistics; useful for logs and Celery task results.
    """
    import time

    t0 = time.perf_counter()
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

    # ------------------------------------------------------------------
    # 1. Build the edge list via a single lateral KNN SQL.
    # ------------------------------------------------------------------
    #
    # For each article A in the window, pull its nearest neighbours B where
    # cosine similarity >= threshold. We dedupe pairs by enforcing A.id < B.id
    # so each undirected edge appears once.
    #
    # ivfflat index on embedding makes this fast even at tens of thousands
    # of rows; the LATERAL join emits ``knn_candidates`` edges per article
    # at most, giving a hard upper bound on memory.

    edge_sql = text(
        """
        SELECT a.id AS a_id, nbr.id AS b_id, nbr.sim AS sim
        FROM news_items a,
             LATERAL (
                 SELECT b.id,
                        1.0 - (b.embedding <=> a.embedding) AS sim
                 FROM news_items b
                 WHERE b.id > a.id
                   AND b.last_seen_at >= :cutoff
                   AND b.embedding IS NOT NULL
                   AND 1.0 - (b.embedding <=> a.embedding) >= :threshold
                 ORDER BY b.embedding <=> a.embedding
                 LIMIT :knn
             ) AS nbr
        WHERE a.last_seen_at >= :cutoff
          AND a.embedding IS NOT NULL
        """
    )

    edges = session.execute(
        edge_sql,
        {"cutoff": cutoff, "threshold": threshold, "knn": knn_candidates},
    ).all()
    logger.info(
        "cluster_recent_articles: edges found (sim ≥ %.2f, window %dd) = %d",
        threshold,
        window_days,
        len(edges),
    )

    # ------------------------------------------------------------------
    # 2. Fetch all articles in the window with title+language for the
    #    overlap filter and representative selection.
    # ------------------------------------------------------------------
    article_rows = session.execute(
        text(
            """
            SELECT id, last_seen_at, title, language
            FROM news_items
            WHERE last_seen_at >= :cutoff
              AND embedding IS NOT NULL
            """
        ),
        {"cutoff": cutoff},
    ).all()
    last_seen_by_id: dict[int, datetime] = {r.id: r.last_seen_at for r in article_rows}
    tokens_by_id: dict[int, set[str]] = {
        r.id: _title_tokens(r.title) for r in article_rows
    }
    lang_by_id: dict[int, str | None] = {r.id: r.language for r in article_rows}
    article_ids = list(last_seen_by_id.keys())
    if not article_ids:
        return ClusteringResult(0, 0, 0, 0, 0, 0, time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # 3. Apply title-overlap filter to suppress template-based chains.
    # ------------------------------------------------------------------
    filtered_edges = []
    for e in edges:
        if _pair_passes_overlap(
            lang_by_id.get(e.a_id),
            lang_by_id.get(e.b_id),
            tokens_by_id.get(e.a_id, set()),
            tokens_by_id.get(e.b_id, set()),
            title_overlap_min,
        ):
            filtered_edges.append(e)

    logger.info(
        "cluster_recent_articles: edges after overlap filter (min=%d) = %d",
        title_overlap_min,
        len(filtered_edges),
    )

    # ------------------------------------------------------------------
    # 4. Union-Find
    # ------------------------------------------------------------------
    uf = _UnionFind()
    for aid in article_ids:
        uf._add(aid)
    for edge in filtered_edges:
        uf.union(edge.a_id, edge.b_id)

    components = uf.components(article_ids)
    non_singletons = [g for g in components.values() if len(g) >= 2]
    largest = max((len(g) for g in components.values()), default=0)

    # ------------------------------------------------------------------
    # 4. Pick representative per component (newest last_seen_at) and
    #    compute similarity_to_rep via a second KNN-ish call. For members
    #    that already appeared as neighbours of the rep (or vice versa),
    #    the similarity is in our edge list; otherwise fall back to a
    #    direct pgvector distance lookup for correctness.
    # ------------------------------------------------------------------
    # Pre-index *filtered* edges keyed as a sorted pair for O(1) lookup.
    # Note: we use filtered_edges here because the pair must have passed the
    # overlap check for us to include their similarity value.
    edge_sim: dict[tuple[int, int], float] = {}
    for e in filtered_edges:
        key = (e.a_id, e.b_id) if e.a_id < e.b_id else (e.b_id, e.a_id)
        edge_sim[key] = float(e.sim)

    update_rows: list[dict] = []  # fed to executemany below
    for members in components.values():
        if len(members) == 1:
            (only_id,) = members
            update_rows.append(
                {
                    "id": only_id,
                    "cluster_rep_id": None,
                    "cluster_size": 1,
                    "similarity_to_rep": None,
                }
            )
            continue

        # Representative: newest. Stable tiebreak on id for determinism.
        rep_id = max(members, key=lambda m: (last_seen_by_id[m], m))
        size = len(members)
        for m in members:
            if m == rep_id:
                sim_val: float | None = 1.0
            else:
                key = (m, rep_id) if m < rep_id else (rep_id, m)
                sim_val = edge_sim.get(key)
                if sim_val is None:
                    # Not directly in edge list (chained through others) —
                    # compute real cosine for the record.
                    sim_val = _direct_similarity(session, m, rep_id)
            update_rows.append(
                {
                    "id": m,
                    "cluster_rep_id": rep_id,
                    "cluster_size": size,
                    "similarity_to_rep": sim_val,
                }
            )

    # ------------------------------------------------------------------
    # 5. Bulk UPDATE
    # ------------------------------------------------------------------
    if update_rows:
        update_sql = text(
            """
            UPDATE news_items
               SET cluster_rep_id    = :cluster_rep_id,
                   cluster_size      = :cluster_size,
                   similarity_to_rep = :similarity_to_rep,
                   updated_at        = NOW()
             WHERE id = :id
            """
        )
        # executemany via execute with list-of-dicts; SQLAlchemy handles batching
        session.execute(update_sql, update_rows)
        session.commit()

    elapsed = time.perf_counter() - t0
    logger.info(
        "cluster_recent_articles: done — considered=%d, edges=%d, "
        "edges_after_overlap=%d, non-singleton clusters=%d, largest=%d, "
        "updated=%d, %.1fs",
        len(article_ids),
        len(edges),
        len(filtered_edges),
        len(non_singletons),
        largest,
        len(update_rows),
        elapsed,
    )
    return ClusteringResult(
        articles_considered=len(article_ids),
        edges_found=len(edges),
        edges_after_overlap=len(filtered_edges),
        non_singleton_clusters=len(non_singletons),
        largest_cluster_size=largest,
        updated_rows=len(update_rows),
        elapsed_sec=elapsed,
    )


def _direct_similarity(session: Session, a_id: int, b_id: int) -> float | None:
    """Fallback cosine similarity between two article embeddings via pgvector."""
    result = session.execute(
        text(
            """
            SELECT 1.0 - (a.embedding <=> b.embedding) AS sim
            FROM news_items a
            JOIN news_items b ON b.id = :b_id
            WHERE a.id = :a_id
              AND a.embedding IS NOT NULL
              AND b.embedding IS NOT NULL
            """
        ),
        {"a_id": a_id, "b_id": b_id},
    ).first()
    return float(result.sim) if result and result.sim is not None else None
