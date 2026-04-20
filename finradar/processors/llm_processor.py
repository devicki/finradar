"""
finradar.processors.llm_processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cloud LLM processor for high-quality summarisation, translation, and structured
metadata extraction from financial news articles.

Supports two providers via an adapter pattern:
  - **Anthropic** (default) — ``claude-opus-4-6``
  - **OpenAI**              — ``gpt-4o-mini``

The active provider is selected from ``settings.llm_provider`` and can be
overridden at construction time.  API clients are instantiated lazily on first
use so the module can be imported without valid API keys present (e.g. during
unit tests that mock the client).

All public methods are ``async``.  The underlying Anthropic and OpenAI SDK calls
are synchronous but are wrapped with ``asyncio.get_event_loop().run_in_executor``
so they do not block the event loop.  This is important because FinRadar's
FastAPI application and Celery workers both run in async contexts.

Prompt caching:
  The Anthropic client uses prompt caching (``cache_control`` on the system
  message) so that repeated calls with the same task type share a cached prefix,
  reducing latency and cost on high-volume pipelines.

Usage::

    from finradar.processors.llm_processor import LLMProcessor

    processor = LLMProcessor()  # uses settings.llm_provider

    summary = await processor.summarize(
        title="Apple beats Q3 earnings estimates",
        content="Apple Inc. reported quarterly revenue of ...",
        language="en",
    )

    korean_title = await processor.translate(
        text="Apple beats Q3 earnings estimates",
        source_lang="en",
        target_lang="ko",
    )

    metadata = await processor.extract_metadata(
        title="NVIDIA soars on AI chip demand",
        content="...",
    )
    # {"tickers": ["NVDA"], "sectors": ["AI", "반도체"]}
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import partial
from typing import TYPE_CHECKING, Any

from finradar.config import get_settings

if TYPE_CHECKING:
    import anthropic
    import openai

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SUMMARIZE_SYSTEM_EN = (
    "You are a senior financial news analyst. "
    "Your task is to produce a concise, factual summary of financial news articles. "
    "Each summary must:\n"
    "  - Be exactly 2–3 sentences long.\n"
    "  - Focus on the key facts, quantitative figures (prices, percentages, earnings), "
    "and the primary market or sector impact.\n"
    "  - Mention relevant ticker symbols (e.g. AAPL, TSLA) if they appear in the article.\n"
    "  - Use clear, professional financial language.\n"
    "  - Avoid speculation or editorial opinion.\n"
    "Return only the summary text — no headings, bullet points, or preamble."
)

_SUMMARIZE_SYSTEM_KO = (
    "당신은 시니어 금융 뉴스 애널리스트입니다. "
    "다음 금융 뉴스 기사를 한국어로 2~3문장 요약해 주세요. "
    "요약에는 핵심 사실, 정량적 수치(주가, 퍼센트, 실적), 주요 시장 또는 섹터 영향이 포함돼야 합니다. "
    "관련 티커 심볼(예: AAPL, 삼성전자)이 있다면 언급하세요. "
    "전문적이고 간결한 금융 언어를 사용하세요. "
    "요약 텍스트만 반환하고, 제목이나 부연 설명은 포함하지 마세요."
)

_TRANSLATE_SYSTEM = (
    "You are a professional financial translator with deep expertise in both "
    "financial markets and multilingual communication. "
    "Translate the provided financial news text accurately and naturally. "
    "Rules:\n"
    "  - Preserve all ticker symbols exactly as-is (e.g. AAPL, TSLA, 005930.KS).\n"
    "  - Preserve numeric values, percentages, and currency symbols unchanged.\n"
    "  - Use standard financial terminology appropriate for the target language.\n"
    "  - Maintain the original tone and factual precision.\n"
    "  - Return only the translated text — no explanations or annotations."
)

_METADATA_SYSTEM = (
    "You are a financial information extraction engine. "
    "Extract structured metadata from the provided financial news article.\n\n"
    "You MUST return a single, valid JSON object with exactly these two keys:\n"
    "  \"tickers\": an array of uppercase stock ticker symbols mentioned "
    "(e.g. [\"AAPL\", \"TSLA\"]). Use empty array [] if none found.\n"
    "  \"sectors\": an array of relevant financial sectors or industry themes "
    "in Korean (e.g. [\"반도체\", \"AI\", \"바이오\", \"에너지\"]) or English if "
    "the article is primarily in English "
    "(e.g. [\"Semiconductors\", \"AI\", \"Energy\"]). Use empty array [] if none found.\n\n"
    "Return ONLY the JSON object. No markdown fences, no explanations, no extra text. "
    "Example output: {\"tickers\": [\"NVDA\", \"AMD\"], \"sectors\": [\"반도체\", \"AI\"]}"
)

_ENRICH_ALL_SYSTEM = (
    "You are a senior financial news analyst and translator. "
    "Given a financial news article, perform ALL of the following tasks in a single response.\n\n"
    "You MUST return a single, valid JSON object with exactly these keys:\n"
    "  \"ai_summary\": A concise 2-3 sentence English summary focusing on key facts, "
    "quantitative figures, and market impact. "
    "IMPORTANT: Only use information explicitly present in the article. "
    "Do NOT add facts, figures, or context not found in the original text. "
    "If the article is too short to summarize, return the original text as-is.\n"
    "  \"translated_title\": The article title translated to Korean. "
    "Preserve ticker symbols and numbers as-is.\n"
    "  \"translated_summary\": The summary (or article content) translated to Korean. "
    "Preserve ticker symbols and numbers as-is.\n"
    "  \"tickers\": An array of uppercase stock ticker symbols mentioned "
    "(e.g. [\"AAPL\", \"TSLA\"]). Use empty array [] if none found.\n"
    "  \"sectors\": An array of relevant financial sectors in Korean "
    "(e.g. [\"반도체\", \"AI\", \"에너지\"]). Use empty array [] if none found.\n\n"
    "Rules:\n"
    "  - Use professional financial language.\n"
    "  - Preserve all ticker symbols, numeric values, percentages, and currency symbols unchanged.\n"
    "  - Return ONLY the JSON object. No markdown fences, no explanations, no extra text.\n\n"
    "Example output:\n"
    "{\"ai_summary\": \"NVIDIA reported Q3 revenue of $35.1B, beating estimates by 12%...\", "
    "\"translated_title\": \"엔비디아, AI 칩 수요 급증으로 3분기 실적 예상치 상회\", "
    "\"translated_summary\": \"엔비디아가 3분기 매출 351억 달러를 기록하며...\", "
    "\"tickers\": [\"NVDA\"], \"sectors\": [\"반도체\", \"AI\"]}"
)

# 한국어 기사 전용 enrich 프롬프트. 이미 한국어라 번역은 불필요하며, 요약도
# 한국어로 생성한다. 응답 JSON 스키마는 영어 프롬프트와 호환되도록 동일한
# 키셋을 유지하되 translated_title / translated_summary는 빈 문자열을 반환한다.
_ENRICH_KO_SYSTEM = (
    "당신은 시니어 금융 뉴스 애널리스트입니다. "
    "주어진 한국어 금융 뉴스 기사에 대해 다음 작업을 단일 응답으로 수행하세요.\n\n"
    "반드시 다음 키를 가진 유효한 JSON 객체 하나만 반환하세요:\n"
    "  \"ai_summary\": 한국어로 2~3문장 요약. 핵심 사실, 정량적 수치(주가, 퍼센트, 실적), "
    "주요 시장/섹터 영향에 집중하세요. "
    "중요: 기사에 명시된 정보만 사용하세요. 없는 사실·수치·맥락을 추가하지 마세요. "
    "기사가 너무 짧으면 원문을 그대로 반환하세요.\n"
    "  \"translated_title\": 빈 문자열 \"\" (한국어 기사이므로 번역 불필요)\n"
    "  \"translated_summary\": 빈 문자열 \"\" (한국어 기사이므로 번역 불필요)\n"
    "  \"tickers\": 기사에서 언급된 대문자 티커 심볼 배열 "
    "(예: [\"AAPL\", \"005930.KS\"]). 없으면 빈 배열 [].\n"
    "  \"sectors\": 한국어 섹터/테마 배열 "
    "(예: [\"반도체\", \"AI\", \"바이오\", \"에너지\"]). 없으면 빈 배열 [].\n\n"
    "규칙:\n"
    "  - 전문적이고 간결한 금융 언어를 사용하세요.\n"
    "  - 티커, 숫자, 퍼센트, 통화 기호는 원문 그대로 유지하세요.\n"
    "  - JSON 객체만 반환하세요. 마크다운, 설명, 추가 텍스트 금지.\n\n"
    "예시 출력:\n"
    "{\"ai_summary\": \"삼성전자가 3분기 매출 79조원을 기록하며 전년 동기 대비 17% 증가했다. "
    "반도체 부문이 실적을 견인했다.\", "
    "\"translated_title\": \"\", \"translated_summary\": \"\", "
    "\"tickers\": [\"005930.KS\"], \"sectors\": [\"반도체\", \"AI\"]}"
)


class LLMProcessor:
    """Cloud LLM processor for summarisation, translation, and metadata extraction.

    Args:
        provider: ``"anthropic"``, ``"openai"``, or ``"grok"``.  Defaults to
                  ``settings.llm_provider``.
    """

    def __init__(self, provider: str | None = None) -> None:
        settings = get_settings()
        self.provider: str = provider or settings.llm_provider
        self._client: anthropic.Anthropic | openai.OpenAI | None = None
        self.logger = logging.getLogger("finradar.processors.llm")

    # ------------------------------------------------------------------
    # Client initialisation (lazy)
    # ------------------------------------------------------------------

    def _get_client(self) -> "anthropic.Anthropic | openai.OpenAI":
        """Instantiate and cache the provider API client on first call."""
        if self._client is not None:
            return self._client

        settings = get_settings()

        if self.provider == "anthropic":
            import anthropic

            if not settings.anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not set. "
                    "Add it to your .env file or environment variables."
                )
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self.logger.info("Anthropic client initialised (model: %s)", settings.anthropic_model)

        elif self.provider == "openai":
            import openai

            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. "
                    "Add it to your .env file or environment variables."
                )
            self._client = openai.OpenAI(api_key=settings.openai_api_key)
            self.logger.info("OpenAI client initialised (model: %s)", settings.openai_model)

        elif self.provider == "grok":
            import openai

            if not settings.grok_api_key:
                raise ValueError(
                    "GROK_API_KEY is not set. "
                    "Add it to your .env file or environment variables."
                )
            self._client = openai.OpenAI(
                api_key=settings.grok_api_key,
                base_url=settings.grok_base_url,
            )
            self.logger.info("Grok client initialised (model: %s)", settings.grok_model)

        else:
            raise ValueError(
                f"Unknown LLM provider: {self.provider!r}. "
                "Accepted values: 'anthropic', 'openai', 'grok'."
            )

        return self._client

    # ------------------------------------------------------------------
    # Low-level provider calls (synchronous, run in executor)
    # ------------------------------------------------------------------

    def _call_anthropic_sync(self, system: str, user_message: str) -> str:
        """Synchronous Anthropic API call.  Must not be called directly in async code."""
        import anthropic

        client: anthropic.Anthropic = self._get_client()  # type: ignore[assignment]
        settings = get_settings()

        message = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},  # prompt caching
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )

        # Extract first text block from content list.
        for block in message.content:
            if block.type == "text":
                return block.text.strip()

        self.logger.warning("Anthropic response contained no text block; returning empty string.")
        return ""

    def _call_openai_sync(self, system: str, user_message: str) -> str:
        """Synchronous OpenAI/Grok API call.  Must not be called directly in async code."""
        import openai

        client: openai.OpenAI = self._get_client()  # type: ignore[assignment]
        settings = get_settings()

        model = settings.active_llm_model()
        response = client.chat.completions.create(
            model=model,
            max_tokens=settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )

        content = response.choices[0].message.content
        if content is None:
            self.logger.warning("OpenAI response returned None content; returning empty string.")
            return ""
        return content.strip()

    async def _call_llm(self, system: str, user_message: str) -> str:
        """Dispatch an LLM request to the configured provider asynchronously.

        The synchronous SDK calls are offloaded to a thread-pool executor so
        they do not block the asyncio event loop.

        Args:
            system:       The system prompt for the LLM.
            user_message: The user-facing content of the request.

        Returns:
            The model's text response, stripped of leading/trailing whitespace.

        Raises:
            RuntimeError: If the API call fails after the SDK's built-in retry.
        """
        loop = asyncio.get_event_loop()

        try:
            if self.provider == "anthropic":
                sync_fn = partial(self._call_anthropic_sync, system, user_message)
            elif self.provider in ("openai", "grok"):
                sync_fn = partial(self._call_openai_sync, system, user_message)
            else:
                sync_fn = partial(self._call_openai_sync, system, user_message)

            return await loop.run_in_executor(None, sync_fn)

        except Exception as exc:
            self.logger.error(
                "LLM call failed (provider=%s): %s", self.provider, exc, exc_info=True
            )
            raise RuntimeError(
                f"LLM call failed via provider {self.provider!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # High-level task methods
    # ------------------------------------------------------------------

    async def summarize(
        self,
        title: str,
        content: str,
        language: str = "en",
    ) -> str:
        """Generate a concise 2–3 sentence summary of a news article.

        Args:
            title:    Article headline.
            content:  Article body text (can be partial if truncated).
            language: Target language for the summary.  ``"ko"`` produces a
                      Korean-language summary; any other value uses English.

        Returns:
            A plain-text 2–3 sentence summary string.

        Raises:
            RuntimeError: If the underlying LLM call fails.
        """
        system = _SUMMARIZE_SYSTEM_KO if language == "ko" else _SUMMARIZE_SYSTEM_EN

        user_message = f"Title: {title}\n\nArticle:\n{content}"

        self.logger.debug(
            "Summarising article (lang=%s, title=%r)", language, title[:80]
        )
        result = await self._call_llm(system, user_message)
        self.logger.debug("Summary produced (%d chars)", len(result))
        return result

    async def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "ko",
    ) -> str:
        """Translate text between languages, preserving financial terminology.

        Args:
            text:        The text to translate.
            source_lang: ISO 639-1 code of the source language.
            target_lang: ISO 639-1 code of the target language.

        Returns:
            The translated text as a plain string.

        Raises:
            RuntimeError: If the underlying LLM call fails.
            ValueError:   If source and target languages are identical.
        """
        if source_lang == target_lang:
            return text  # No-op — same language, return unchanged.

        lang_names = {
            "en": "English",
            "ko": "Korean",
            "ja": "Japanese",
            "zh": "Chinese",
            "de": "German",
            "fr": "French",
        }
        source_name = lang_names.get(source_lang, source_lang.upper())
        target_name = lang_names.get(target_lang, target_lang.upper())

        user_message = (
            f"Translate the following text from {source_name} to {target_name}:\n\n{text}"
        )

        self.logger.debug(
            "Translating %d chars from %s to %s", len(text), source_lang, target_lang
        )
        result = await self._call_llm(_TRANSLATE_SYSTEM, user_message)
        self.logger.debug("Translation produced (%d chars)", len(result))
        return result

    async def extract_metadata(self, title: str, content: str) -> dict[str, Any]:
        """Extract structured tickers and sectors from a news article.

        Uses the LLM to identify stock tickers and relevant industry sectors
        mentioned in the article.  Returns safe defaults on parse failure so
        the pipeline is never blocked by malformed LLM output.

        Args:
            title:   Article headline.
            content: Article body text.

        Returns:
            dict with keys:

            * ``tickers`` — list of uppercase ticker symbols (e.g. ``["AAPL", "TSLA"]``)
            * ``sectors`` — list of sector/theme strings (e.g. ``["AI", "반도체"]``)

            Both lists default to ``[]`` if the LLM returns no matches or if
            JSON parsing fails.

        Raises:
            RuntimeError: If the underlying LLM call fails at the network level.
        """
        user_message = (
            f"Title: {title}\n\n"
            f"Article (first 1500 chars):\n{content[:1500]}"
        )

        self.logger.debug(
            "Extracting metadata from article: %r", title[:80]
        )

        raw_response = await self._call_llm(_METADATA_SYSTEM, user_message)

        # ------------------------------------------------------------------
        # Parse JSON — the LLM should return a bare JSON object, but may
        # wrap it in markdown fences.  We try two strategies: direct parse,
        # then regex extraction of the first JSON object.
        # ------------------------------------------------------------------
        parsed = self._parse_metadata_response(raw_response, title)
        self.logger.debug(
            "Metadata extracted — tickers=%s, sectors=%s",
            parsed.get("tickers"),
            parsed.get("sectors"),
        )
        return parsed

    def _parse_metadata_response(
        self, raw: str, title: str = ""
    ) -> dict[str, Any]:
        """Parse the LLM's JSON metadata response with graceful fallback.

        Strategy:
        1. Try ``json.loads()`` on the stripped response directly.
        2. If that fails, extract the first ``{...}`` block with a regex and
           try again.
        3. If all parsing fails, log a warning and return safe empty defaults.

        Args:
            raw:   Raw text response from the LLM.
            title: Article title used only in log messages.

        Returns:
            dict with ``tickers`` (list[str]) and ``sectors`` (list[str]).
        """
        empty: dict[str, Any] = {"tickers": [], "sectors": []}

        if not raw:
            return empty

        def _normalise(data: Any) -> dict[str, Any]:
            """Ensure the parsed dict has the expected shape."""
            if not isinstance(data, dict):
                return empty
            tickers = data.get("tickers", [])
            sectors = data.get("sectors", [])
            # Coerce to lists of strings.
            if not isinstance(tickers, list):
                tickers = []
            if not isinstance(sectors, list):
                sectors = []
            tickers = [str(t).upper().strip() for t in tickers if t]
            sectors = [str(s).strip() for s in sectors if s]
            return {"tickers": tickers, "sectors": sectors}

        # Strategy 1: direct parse
        try:
            return _normalise(json.loads(raw.strip()))
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract first {...} block
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                return _normalise(json.loads(match.group()))
            except json.JSONDecodeError:
                pass

        self.logger.warning(
            "Failed to parse metadata JSON for article %r. Raw response (first 200): %r",
            title[:60],
            raw[:200],
        )
        return empty

    # ------------------------------------------------------------------
    # Combined enrichment (single LLM call)
    # ------------------------------------------------------------------

    async def enrich_article(
        self,
        title: str,
        content: str,
        language: str = "en",
    ) -> dict[str, Any]:
        """Summarize, translate, and extract metadata in a single LLM call.

        Branches by article language:
          - Korean articles use ``_ENRICH_KO_SYSTEM`` — Korean summary, no
            translation (translated_* fields return empty).
          - All other languages use ``_ENRICH_ALL_SYSTEM`` — English summary
            plus Korean translation.

        Returns a dict with keys: ai_summary, translated_title,
        translated_summary, tickers, sectors.  Missing fields default
        to empty string / empty list.

        Args:
            title:    Article headline.
            content:  Article body text.
            language: Source language of the article.

        Raises:
            RuntimeError: If the underlying LLM call fails.
        """
        if language == "ko":
            system_prompt = _ENRICH_KO_SYSTEM
            user_message = (
                f"제목: {title}\n\n"
                f"기사:\n{content[:2000]}"
            )
        else:
            system_prompt = _ENRICH_ALL_SYSTEM
            user_message = (
                f"Title: {title}\n\n"
                f"Article:\n{content[:2000]}"
            )
            if language != "en":
                user_message += f"\n\nNote: The source language is '{language}'."

        self.logger.debug(
            "Enriching article in single call (lang=%s): %r", language, title[:80]
        )
        raw_response = await self._call_llm(system_prompt, user_message)
        parsed = self._parse_enrich_response(raw_response, title)
        self.logger.debug(
            "Enrichment complete — summary=%d chars, tickers=%s, sectors=%s",
            len(parsed.get("ai_summary", "")),
            parsed.get("tickers"),
            parsed.get("sectors"),
        )
        return parsed

    def _parse_enrich_response(
        self, raw: str, title: str = ""
    ) -> dict[str, Any]:
        """Parse the combined enrichment JSON response with graceful fallback."""
        empty: dict[str, Any] = {
            "ai_summary": "",
            "translated_title": "",
            "translated_summary": "",
            "tickers": [],
            "sectors": [],
        }

        if not raw:
            return empty

        def _normalise(data: Any) -> dict[str, Any]:
            if not isinstance(data, dict):
                return empty
            result = {
                "ai_summary": str(data.get("ai_summary", "")).strip(),
                "translated_title": str(data.get("translated_title", "")).strip(),
                "translated_summary": str(data.get("translated_summary", "")).strip(),
                "tickers": [],
                "sectors": [],
            }
            tickers = data.get("tickers", [])
            sectors = data.get("sectors", [])
            if isinstance(tickers, list):
                result["tickers"] = [str(t).upper().strip() for t in tickers if t]
            if isinstance(sectors, list):
                result["sectors"] = [str(s).strip() for s in sectors if s]
            return result

        # Strategy 1: direct parse
        try:
            return _normalise(json.loads(raw.strip()))
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract first {...} block (greedy for nested JSON)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return _normalise(json.loads(match.group()))
            except json.JSONDecodeError:
                pass

        self.logger.warning(
            "Failed to parse enrichment JSON for article %r. Raw (first 300): %r",
            title[:60],
            raw[:300],
        )
        return empty

    # ------------------------------------------------------------------
    # Convenience class-method constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls) -> "LLMProcessor":
        """Construct an LLMProcessor from application settings.

        Uses ``settings.llm_provider`` to select the provider.
        """
        settings = get_settings()
        return cls(provider=settings.llm_provider)
