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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session


logger = logging.getLogger("finradar.clustering")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_DAYS: int = 7
# Baseline cosine used for the initial KNN pull in SQL.  Same-language pairs
# are filtered further in Python using ``DEFAULT_SAME_LANG_COSINE`` because
# the multilingual embedding compresses Korean news into a dense region —
# 0.80 catches unrelated template chains in KR corpora.
DEFAULT_COSINE_THRESHOLD: float = 0.80
DEFAULT_SAME_LANG_COSINE: float = 0.85
DEFAULT_KNN_CANDIDATES: int = 30
DEFAULT_TITLE_OVERLAP_MIN: int = 2     # required shared title tokens (same-lang)
DEFAULT_TITLE_OVERLAP_RATIO: float = 0.30  # shared_tokens / min(|A|,|B|) floor
DEFAULT_MIN_BODY_JACCARD: float = 0.15     # summary bigram Jaccard floor; 0 disables

# Tokens shorter than this are treated as noise (particles, prepositions, digits).
_MIN_TOKEN_LEN: int = 2

# Word boundary for title/body tokenisation — whitespace + punctuation commonly
# found in KO/EN headlines (…, [], (), quotes, dashes, etc.).
_TOKEN_SPLIT = re.compile(r"[\s,.…·!?;:'\"\[\]\(\)\-—–/]+")
_TITLE_TOKEN_SPLIT = _TOKEN_SPLIT  # backward-compat alias

# Bracket prefix at the start of a headline (e.g. "[포토] ...", "[美특징주] ...").
# Korean press uses these as section markers, not content — they chain many
# unrelated stories when left as plain tokens.
_BRACKET_PREFIX_RE = re.compile(r"^[\[\(](?:[^\[\](),]+)[\]\)]\s*")

# Trailing attribution suffix: "- 증권사명" or "-증권사" at the end.  Korean
# target-price headlines end with these and the broker name itself is noise.
_ATTRIBUTION_SUFFIX_RE = re.compile(r"\s*[-–—]\s*[^\s\-–—]{1,10}\s*$")

# Korean financial boilerplate stopwords.  Pure markers, period labels, movement
# verbs, and market-state words that appear across many unrelated stories.
# Brand / ticker / sector words (삼성전자, SK하이닉스, 반도체, AI, …) are
# intentionally kept — those are the real clustering signal.
_KR_STOPWORDS: frozenset[str] = frozenset(
    {
        # Bracket-prefix residue after stripping the brackets themselves
        "포토", "속보", "단독", "특징주", "美특징주", "개장전", "개장",
        "마감", "르포", "영상", "사설", "인사", "카드뉴스", "뉴스토리",
        # Period labels
        "1분기", "2분기", "3분기", "4분기", "상반기", "하반기", "분기",
        "전년", "동기", "올해", "작년", "연말", "연초",
        # Finance KPIs — generic labels
        "실적", "영업익", "영업이익", "매출", "순익", "순이익", "순손실",
        "주가", "종가", "시가", "거래량", "시총", "시가총액",
        # Scale superlatives
        "사상", "최대", "최고", "최저", "최고치", "최저치", "사상최대",
        "사상최고", "역대", "최고가",
        # Movement verbs/nouns (generic)
        "돌파", "상승", "하락", "급등", "급락", "반등", "보합", "강세",
        "약세", "상향", "하향", "확대", "축소", "유지", "지속",
        # News/market state
        "연속", "공개", "발표", "예상", "전망", "기록", "집계",
        # Common fillers
        "기사", "리포트", "보고서", "기자", "관련", "대비",
    }
)

_EN_STOPWORDS: frozenset[str] = frozenset(
    {
        # Common English headline fillers (length >= 2 already passes min-len)
        "the", "and", "for", "with", "from", "about", "into", "over",
        "this", "that", "will", "says", "said", "amid",
        # Period labels
        "q1", "q2", "q3", "q4", "1q", "2q", "3q", "4q",
        "h1", "h2", "fy", "year", "years", "quarter", "quarterly",
        # Finance KPIs
        "earnings", "revenue", "revenues", "profit", "profits",
        "loss", "losses", "sales", "margin", "margins",
        "stock", "stocks", "share", "shares", "price", "prices",
        "close", "open", "high", "low",
        # Movement verbs
        "surge", "surges", "surged", "plunge", "plunges", "plunged",
        "rally", "rallies", "rallied", "slump", "slumped",
        "gain", "gains", "gained", "fall", "falls", "fell",
        "jump", "jumps", "jumped", "drop", "drops", "dropped",
        "rise", "rises", "rising", "rose", "climb", "climbs", "climbed",
        "surges", "slides",
        # News/market state
        "breaking", "update", "updates", "report", "reports", "reported",
        "according", "via", "source", "sources",
    }
)


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
    # Per-reject-reason counts so dry-runs and logs can tell which filter is
    # doing the work. Populated by :func:`cluster_recent_articles` when the
    # extended filter chain is active.
    rejected_by_reason: dict[str, int] = field(default_factory=dict)
    # Dry-run payload — when ``write=False`` the clusterer attaches the list
    # of per-article assignment dicts it *would* have written so calling
    # scripts can inspect the new cluster-size distribution without touching
    # the DB.  Empty in the normal write-through path.
    assignments: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Title-overlap helpers
# ---------------------------------------------------------------------------


def _clean_title(title: str | None) -> str:
    """Strip bracket prefixes and trailing attribution from a headline.

    Examples
    --------
    >>> _clean_title("[포토] 코스피 상승세")
    '코스피 상승세'
    >>> _clean_title("농심, 목표가 50만원으로 하향-다올")
    '농심, 목표가 50만원으로 하향'
    """
    if not title:
        return ""
    cleaned = title
    # Some headlines stack multiple bracket prefixes; strip all of them.
    while True:
        new = _BRACKET_PREFIX_RE.sub("", cleaned, count=1)
        if new == cleaned:
            break
        cleaned = new
    cleaned = _ATTRIBUTION_SUFFIX_RE.sub("", cleaned)
    return cleaned


def _title_tokens(
    title: str | None,
    language: str | None = None,
    *,
    apply_stopwords: bool = True,
) -> set[str]:
    """Extract content tokens (length ≥ 2) from a headline.

    Lowercased so English "Fed" and "fed" match; Korean text unaffected.
    Bracket prefixes and trailing attribution are removed first, then
    language-specific boilerplate stopwords are dropped when
    ``apply_stopwords`` is true.
    """
    if not title:
        return set()
    cleaned = _clean_title(title)
    tokens = {
        tok.lower()
        for tok in _TOKEN_SPLIT.split(cleaned)
        if len(tok) >= _MIN_TOKEN_LEN
    }
    if not apply_stopwords:
        return tokens
    lang = (language or "").lower()
    if lang == "ko":
        return tokens - _KR_STOPWORDS
    if lang == "en":
        return tokens - _EN_STOPWORDS
    # Unknown/other language — apply the union so both lists filter common
    # noise while not over-filtering for languages we don't curate.
    return tokens - (_KR_STOPWORDS | _EN_STOPWORDS)


def _body_shingles(text_body: str | None) -> set[tuple[str, str]]:
    """Return word-level 2-shingles (bigrams) of a body/summary string.

    Shingles give much better discrimination than unigrams for "same story"
    detection because finance stopwords rarely co-occur in the same order
    across unrelated stories. Caller already decides whether both sides of
    the pair are worth Jaccard'ing (body Jaccard only runs on same-lang).
    """
    if not text_body:
        return set()
    tokens = [
        tok.lower() for tok in _TOKEN_SPLIT.split(text_body) if len(tok) >= _MIN_TOKEN_LEN
    ]
    if len(tokens) < 2:
        return set()
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _body_jaccard(
    a_shingles: set[tuple[str, str]], b_shingles: set[tuple[str, str]]
) -> float:
    """Jaccard similarity between two bigram sets; 0 when either is empty."""
    if not a_shingles or not b_shingles:
        return 0.0
    inter = len(a_shingles & b_shingles)
    if inter == 0:
        return 0.0
    return inter / float(len(a_shingles | b_shingles))


def _tickers_conflict(
    a_tickers: list[str] | None, b_tickers: list[str] | None
) -> bool:
    """True iff both sides mention tickers AND share zero of them.

    Strong negative signal — different companies almost always means
    different stories, even when embeddings rhyme. Safely skipped when
    either side has no ticker data (avoids penalising un-enriched rows).
    """
    if not a_tickers or not b_tickers:
        return False
    return not (set(t.upper() for t in a_tickers) & set(t.upper() for t in b_tickers))


# ---------------------------------------------------------------------------
# Edge filter
# ---------------------------------------------------------------------------


def _pair_passes_filters(
    *,
    lang_a: str | None,
    lang_b: str | None,
    cosine: float,
    same_lang_cosine: float,
    tokens_a: set[str],
    tokens_b: set[str],
    min_overlap: int,
    min_overlap_ratio: float,
    tickers_a: list[str] | None,
    tickers_b: list[str] | None,
    shingles_a: set[tuple[str, str]] | None,
    shingles_b: set[tuple[str, str]] | None,
    min_body_jaccard: float,
) -> tuple[bool, str | None]:
    """Combined edge filter. Returns (pass, reject_reason).

    Chain (order chosen so cheaper checks reject first):
      1. Ticker conflict — both mention tickers but share none
      2. Same-language cosine floor (stricter than SQL baseline)
      3. Title-token overlap count + ratio (same-lang only)
      4. Body-text bigram Jaccard floor (same-lang only)
    Cross-language pairs skip the text-overlap / body-Jaccard checks since
    tokens don't transfer across scripts.
    """
    # Hard ticker rejection applies regardless of language: a Bloomberg/한경
    # pair talking about AAPL vs TSLA is clearly different stories even with
    # very high cosine.
    if _tickers_conflict(tickers_a, tickers_b):
        return False, "ticker_conflict"

    same_lang = (lang_a or "") == (lang_b or "")

    if same_lang and cosine < same_lang_cosine:
        return False, "cosine_below_same_lang"

    if same_lang:
        shared = tokens_a & tokens_b
        if len(shared) < min_overlap:
            return False, "overlap_count"
        smaller = min(len(tokens_a), len(tokens_b)) or 1
        if len(shared) / smaller < min_overlap_ratio:
            return False, "overlap_ratio"

        # Body Jaccard only meaningful when BOTH sides have body text. Skip
        # the check when either summary is empty (otherwise we'd over-reject
        # short headlines / RSS-without-body rows that the overlap filter
        # already decided are legitimately connected).
        if min_body_jaccard > 0 and shingles_a and shingles_b:
            jac = _body_jaccard(shingles_a, shingles_b)
            if jac < min_body_jaccard:
                return False, "body_jaccard"

    return True, None


def _pair_passes_overlap(
    lang_a: str | None,
    lang_b: str | None,
    tokens_a: set[str],
    tokens_b: set[str],
    min_overlap: int,
) -> bool:
    """Legacy overlap-only filter. Kept for backwards compatibility — the
    production path now goes through :func:`_pair_passes_filters` which
    chains cosine / overlap / Jaccard / ticker-conflict together.
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
    same_lang_cosine: float = DEFAULT_SAME_LANG_COSINE,
    knn_candidates: int = DEFAULT_KNN_CANDIDATES,
    title_overlap_min: int = DEFAULT_TITLE_OVERLAP_MIN,
    title_overlap_ratio: float = DEFAULT_TITLE_OVERLAP_RATIO,
    min_body_jaccard: float = DEFAULT_MIN_BODY_JACCARD,
    apply_stopwords: bool = True,
    write: bool = True,
) -> ClusteringResult:
    """Cluster all articles inside the recency window.

    Intended to be called from a Celery task with a synchronous SQLAlchemy
    Session (see ``collection_tasks.SyncSessionLocal``).

    Args:
        session:              Sync SQLAlchemy session (bound to psycopg2 engine).
        window_days:          Articles with ``last_seen_at`` within this many
                              days participate. Older clusters stay as-is.
        threshold:            Baseline cosine for the SQL KNN pull. Cross-
                              language pairs use this value directly.
        same_lang_cosine:     Tighter cosine floor applied to same-language
                              pairs in Python. The multilingual embedding
                              compresses Korean news into a dense region, so
                              0.85 suppresses weak chains without losing
                              cross-lingual matches (which stay at ``threshold``).
        knn_candidates:       Per-article neighbour cap. Higher → denser edges,
                              more CPU. 30 is plenty for news dedup.
        title_overlap_min:    Required number of shared title tokens for
                              same-language pairs. Raised from 1 → 2 to break
                              Korean template-chain over-clusters.
        title_overlap_ratio:  Shared-tokens / min(|A|,|B|) floor — guards
                              against "2 shared boilerplate words out of 20".
        min_body_jaccard:     Minimum 2-shingle Jaccard on the article
                              summary/body. Catches near-identical titles
                              that actually describe different stories.
                              Set to 0 to disable.
        apply_stopwords:      When true, KR + EN finance boilerplate stopwords
                              are dropped from the title token set.
        write:                When false, run the pipeline without mutating
                              ``news_items`` (used by dry-run scripts).

    Returns:
        Summary statistics (including per-reject-reason counts).
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
    # 2. Fetch all articles in the window with title/language/summary and
    #    the structured-tag columns used by the ticker-conflict guard.
    # ------------------------------------------------------------------
    article_rows = session.execute(
        text(
            """
            SELECT id, last_seen_at, title, language, summary, ai_summary,
                   tickers
            FROM news_items
            WHERE last_seen_at >= :cutoff
              AND embedding IS NOT NULL
            """
        ),
        {"cutoff": cutoff},
    ).all()
    last_seen_by_id: dict[int, datetime] = {r.id: r.last_seen_at for r in article_rows}
    lang_by_id: dict[int, str | None] = {r.id: r.language for r in article_rows}
    tokens_by_id: dict[int, set[str]] = {
        r.id: _title_tokens(r.title, r.language, apply_stopwords=apply_stopwords)
        for r in article_rows
    }
    tickers_by_id: dict[int, list[str] | None] = {
        r.id: list(r.tickers) if r.tickers else None for r in article_rows
    }
    # Body shingles are only materialised when Jaccard filter is active — the
    # body-text pull can be large and computing shingles on skipped rows is
    # wasted work in ``write=False`` dry-runs that turn the check off.
    shingles_by_id: dict[int, set[tuple[str, str]]] = {}
    if min_body_jaccard > 0:
        for r in article_rows:
            body = (r.summary or r.ai_summary or "").strip()
            shingles_by_id[r.id] = _body_shingles(body)

    article_ids = list(last_seen_by_id.keys())
    if not article_ids:
        return ClusteringResult(
            articles_considered=0,
            edges_found=0,
            edges_after_overlap=0,
            non_singleton_clusters=0,
            largest_cluster_size=0,
            updated_rows=0,
            elapsed_sec=time.perf_counter() - t0,
            rejected_by_reason={},
        )

    # ------------------------------------------------------------------
    # 3. Chain: ticker-conflict → same-lang cosine → overlap → body Jaccard.
    # ------------------------------------------------------------------
    filtered_edges = []
    rejected_by_reason: dict[str, int] = defaultdict(int)
    for e in edges:
        ok, reason = _pair_passes_filters(
            lang_a=lang_by_id.get(e.a_id),
            lang_b=lang_by_id.get(e.b_id),
            cosine=float(e.sim),
            same_lang_cosine=same_lang_cosine,
            tokens_a=tokens_by_id.get(e.a_id, set()),
            tokens_b=tokens_by_id.get(e.b_id, set()),
            min_overlap=title_overlap_min,
            min_overlap_ratio=title_overlap_ratio,
            tickers_a=tickers_by_id.get(e.a_id),
            tickers_b=tickers_by_id.get(e.b_id),
            shingles_a=shingles_by_id.get(e.a_id),
            shingles_b=shingles_by_id.get(e.b_id),
            min_body_jaccard=min_body_jaccard,
        )
        if ok:
            filtered_edges.append(e)
        else:
            rejected_by_reason[reason or "unknown"] += 1

    logger.info(
        "cluster_recent_articles: edges after filter chain "
        "(same_lang_cos=%.2f, overlap_min=%d, overlap_ratio=%.2f, "
        "body_jac=%.2f) = %d (rejected by: %s)",
        same_lang_cosine,
        title_overlap_min,
        title_overlap_ratio,
        min_body_jaccard,
        len(filtered_edges),
        dict(rejected_by_reason),
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
    # 5. Bulk UPDATE — skipped when ``write=False`` (dry-run scripts).
    # ------------------------------------------------------------------
    if write and update_rows:
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
        len(update_rows) if write else 0,
        elapsed,
    )
    return ClusteringResult(
        articles_considered=len(article_ids),
        edges_found=len(edges),
        edges_after_overlap=len(filtered_edges),
        non_singleton_clusters=len(non_singletons),
        largest_cluster_size=largest,
        updated_rows=len(update_rows) if write else 0,
        elapsed_sec=elapsed,
        rejected_by_reason=dict(rejected_by_reason),
        assignments=[] if write else list(update_rows),
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
