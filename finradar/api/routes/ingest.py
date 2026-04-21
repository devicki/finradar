"""
finradar.api.routes.ingest
~~~~~~~~~~~~~~~~~~~~~~~~~~

Ad-hoc URL ingestion endpoint.

Accepts any publicly reachable URL (HTML page or PDF), extracts the article
body, and inserts the result into ``news_items`` so it flows through the
existing sentiment + embedding + LLM pipeline.

Scope (M1 per the design review):
  * Public pages only — login-gated URLs return a ``login_required`` status
    with a diagnostic hint instead of silently failing.
  * Supports text/html (trafilatura) and application/pdf (pdfplumber).
  * Best-effort handling of legacy Korean encodings (EUC-KR, CP949) via
    ``httpx``'s auto-detect, which falls back to the HTTP ``Content-Type``
    charset and then to chardet.

The route is intentionally synchronous (from the user's POV): we commit the
row inside the request so the caller can immediately navigate to the article
detail page.  Downstream enrichment (sentiment / embedding / LLM) remains
asynchronous via the existing Celery pipeline.
"""
from __future__ import annotations

import io
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from finradar.api.deps import DbSession
from finradar.collectors.base import CollectedArticle
from finradar.models import NewsItem


logger = logging.getLogger("finradar.api.ingest")

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class IngestUrlRequest(BaseModel):
    url: str = Field(..., min_length=8, description="Public http(s) URL to ingest")
    language: str | None = Field(
        default=None,
        description="Override language tag (default: auto from HTML lang attr or 'ko')",
    )
    force_pdf: bool = Field(
        default=False,
        description="Treat the response as PDF even if Content-Type suggests otherwise",
    )


class IngestUrlResponse(BaseModel):
    status: str = Field(..., description="ok | login_required | unsupported | error")
    news_id: int | None = None
    title: str | None = None
    extracted_len: int = 0
    content_type: str | None = None
    message: str | None = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_FETCH_TIMEOUT = httpx.Timeout(20.0, connect=10.0)

# Login wall heuristics — cheap signals we can check without a full parse.
_LOGIN_PATH_RE = re.compile(r"(^|/)(login|signin|sign_in|auth/login)(\.|/|\?|$)", re.IGNORECASE)
_LOGIN_REDIRECT_RE = re.compile(
    r"""(?:document\.location\.href\s*=|window\.location\s*=|meta\s+http-equiv=["']refresh["']).{0,200}login""",
    re.IGNORECASE | re.DOTALL,
)
_KOREAN_LOGIN_KEYWORDS = ("로그인", "회원가입이 필요", "인증이 필요")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_login_wall(final_url: str, content_type: str, body: str) -> str | None:
    """Return a human-readable reason if the fetch clearly hit a login wall.

    The heuristic is intentionally conservative — false negatives (pages we
    think are OK but actually require login) are preferred over false positives
    (pages we wrongly reject). On a negative the caller proceeds with extraction
    and may still end up with a short/empty body, which is surfaced separately.
    """
    if _LOGIN_PATH_RE.search(final_url):
        return f"Redirected to a login URL: {final_url}"
    if content_type.startswith("text/html"):
        if len(body) < 6_000 and _LOGIN_REDIRECT_RE.search(body):
            return "JavaScript redirect to login detected in small HTML body"
        if len(body) < 3_000 and any(kw in body for kw in _KOREAN_LOGIN_KEYWORDS):
            return "Login-required Korean keywords detected in small HTML body"
    return None


def _extract_html_metadata(html: str) -> tuple[str | None, str | None, str | None]:
    """Return (title, author, iso_date) best-effort via trafilatura.

    All three can be None — we fall back to HTML ``<title>`` and "no author"
    upstream.
    """
    try:
        import trafilatura  # noqa: PLC0415

        meta = trafilatura.extract_metadata(html)
        if meta is None:
            return None, None, None
        return (
            getattr(meta, "title", None),
            getattr(meta, "author", None),
            getattr(meta, "date", None),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("trafilatura metadata extraction failed: %s", exc)
        return None, None, None


def _fallback_title_from_html(html: str) -> str | None:
    """Parse the <title> tag as a last-resort title."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def _looks_like_generic_title(title: str) -> bool:
    """Heuristic: titles with many pipes look like site chrome, not an article.

    Legacy Korean sites often render ``<title>메뉴 | 섹션 | 사이트명</title>``
    where the actual article headline is only inside the body. When the page
    metadata title is clearly generic, the first meaningful line of the
    extracted body is usually a better article title.
    """
    if not title:
        return True
    t = title.strip()
    # Many separator chars → template-style
    seps = sum(t.count(ch) for ch in "|·»›")
    if seps >= 2:
        return True
    # Very short generic site names
    if len(t) <= 10:
        return True
    return False


def _first_meaningful_line(text: str, min_len: int = 10, max_len: int = 200) -> str | None:
    """First non-empty line of *text* whose length is within [min_len, max_len]."""
    for line in text.splitlines():
        line = line.strip()
        if min_len <= len(line) <= max_len:
            return line
    return None


def _detect_language_from_html(html: str) -> str | None:
    """Read <html lang='ko'> attribute if present."""
    m = re.search(r"<html[^>]*\blang=[\"']([a-zA-Z\-]+)[\"']", html, re.IGNORECASE)
    return m.group(1).split("-")[0].lower() if m else None


def _extract_html_body(html: str) -> str:
    """Run trafilatura with recall-leaning settings (Korean legacy pages)."""
    import trafilatura  # noqa: PLC0415

    body = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=False,
        favor_recall=True,
    )
    return body or ""


def _extract_pdf_body(pdf_bytes: bytes) -> tuple[str, str | None]:
    """Extract text + best-effort title from a PDF byte stream."""
    import pdfplumber  # noqa: PLC0415

    parts: list[str] = []
    title: str | None = None
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        # Metadata title if the PDF actually sets it
        try:
            title = (pdf.metadata or {}).get("Title") or None
            if title:
                title = title.strip() or None
        except Exception:  # noqa: BLE001
            title = None

        for page in pdf.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)

    full = "\n".join(parts).strip()

    # Fallback title: first non-empty line of page 1
    if not title and full:
        for line in full.splitlines():
            line = line.strip()
            if 10 <= len(line) <= 200:
                title = line
                break

    return full, title


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/url",
    response_model=IngestUrlResponse,
    status_code=status.HTTP_200_OK,
    summary="Ad-hoc URL ingestion (HTML or PDF)",
    description=(
        "Fetch a public URL, extract the article body (trafilatura for HTML, "
        "pdfplumber for PDF), and insert it into `news_items` with "
        "`source_type='url_report'`. Downstream sentiment / embedding / LLM "
        "enrichment runs automatically via the existing Celery pipeline. "
        "Login-gated URLs return `status='login_required'` with a diagnostic "
        "message — session cookie support is planned for M2."
    ),
)
async def ingest_url(request: IngestUrlRequest, db: DbSession) -> IngestUrlResponse:
    url = request.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL must start with http:// or https://",
        )

    # ------------------------------------------------------------------
    # 1. Already ingested? Short-circuit so repeated paste-ins are idempotent.
    # ------------------------------------------------------------------
    existing = (
        await db.execute(select(NewsItem).where(NewsItem.url == url))
    ).scalar_one_or_none()
    if existing is not None:
        return IngestUrlResponse(
            status="ok",
            news_id=existing.id,
            title=existing.title,
            extracted_len=len(existing.summary or ""),
            content_type=existing.source_type,
            message="URL already ingested — returning existing row.",
        )

    # ------------------------------------------------------------------
    # 2. Fetch
    # ------------------------------------------------------------------
    try:
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
            headers={
                "User-Agent": _BROWSER_UA,
                "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
            },
        ) as client:
            resp = await client.get(url)
    except httpx.HTTPError as exc:
        return IngestUrlResponse(
            status="error",
            message=f"HTTP fetch failed: {exc}",
        )

    if resp.status_code in (401, 403):
        return IngestUrlResponse(
            status="login_required",
            message=f"HTTP {resp.status_code} — this URL requires authentication.",
        )
    if resp.status_code >= 400:
        return IngestUrlResponse(
            status="error",
            message=f"HTTP {resp.status_code} — fetch rejected by origin.",
        )

    final_url = str(resp.url)
    content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    if request.force_pdf:
        content_type = "application/pdf"

    # ------------------------------------------------------------------
    # 3. Branch by content type
    # ------------------------------------------------------------------
    extracted_body: str
    title: str | None = None
    author: str | None = None
    date_str: str | None = None
    detected_lang: str | None = None

    if "pdf" in content_type:
        try:
            extracted_body, title = _extract_pdf_body(resp.content)
        except Exception as exc:  # noqa: BLE001
            return IngestUrlResponse(
                status="error",
                content_type=content_type,
                message=f"PDF extraction failed: {exc}",
            )
        detected_lang = request.language or "ko"
    elif content_type.startswith("text/html") or content_type == "":
        # httpx auto-decodes using response charset → resp.text handles EUC-KR etc.
        html = resp.text

        # Early reject login walls so we don't write noise into the DB.
        login_reason = _detect_login_wall(final_url, content_type or "text/html", html)
        if login_reason:
            return IngestUrlResponse(
                status="login_required",
                content_type=content_type or "text/html",
                message=login_reason,
            )

        extracted_body = _extract_html_body(html)
        if not extracted_body or len(extracted_body) < 40:
            return IngestUrlResponse(
                status="error",
                content_type=content_type or "text/html",
                extracted_len=len(extracted_body),
                message=(
                    "Extraction produced an empty or near-empty body. The page may be "
                    "JavaScript-rendered, paywalled, or under an unsupported layout."
                ),
            )
        meta_title, author, date_str = _extract_html_metadata(html)
        # Prefer the extracted body's first meaningful line when the
        # metadata / <title> tag value looks like site-chrome navigation
        # (common on Korean brokerage / news portal pages).
        candidate_title = meta_title or _fallback_title_from_html(html)
        body_title = _first_meaningful_line(extracted_body)
        if body_title and (
            not candidate_title or _looks_like_generic_title(candidate_title)
        ):
            title = body_title
        else:
            title = candidate_title or body_title
        detected_lang = request.language or _detect_language_from_html(html) or "ko"
    else:
        return IngestUrlResponse(
            status="unsupported",
            content_type=content_type,
            message=f"Content-type {content_type!r} is not supported (only HTML / PDF).",
        )

    if not title:
        # Absolute worst case: use the URL path as title
        title = final_url.rsplit("/", 1)[-1][:150] or "(untitled)"

    # Parse date_str → datetime if trafilatura gave one
    published_at: datetime | None = None
    if date_str:
        try:
            published_at = datetime.fromisoformat(date_str)
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)
        except ValueError:
            published_at = None
    if published_at is None:
        published_at = datetime.now(tz=timezone.utc)

    article = CollectedArticle(
        title=title[:500],
        url=final_url,
        source_url="url_ingest",
        source_type="url_report",
        language=detected_lang,
        summary=extracted_body,
        published_at=published_at,
        tickers=[],
        sectors=[],
        raw_data={
            "url_ingest": {
                "requested_url": url,
                "final_url": final_url,
                "content_type": content_type or "text/html",
                "author": author,
            }
        },
    )

    # ------------------------------------------------------------------
    # 4. Persist via the shared upsert logic.  We need the sync session
    # factory from the tasks module because _upsert_articles expects it.
    # ------------------------------------------------------------------
    from finradar.tasks.collection_tasks import SyncSessionLocal, _upsert_articles  # noqa: PLC0415

    with SyncSessionLocal() as session:
        inserted, updated = _upsert_articles(session, [article])
        session.commit()

        # Look up the row we just wrote to return its id.
        row = session.execute(
            select(NewsItem).where(NewsItem.url == final_url)
        ).scalar_one_or_none()

    if row is None:
        return IngestUrlResponse(
            status="error",
            message="Upsert completed but the row could not be fetched back.",
        )

    return IngestUrlResponse(
        status="ok",
        news_id=row.id,
        title=row.title,
        extracted_len=len(extracted_body),
        content_type=content_type or "text/html",
        message=(
            f"inserted={inserted} updated={updated} — downstream sentiment / "
            "embedding / LLM enrichment will run on the next pipeline tick."
        ),
    )
