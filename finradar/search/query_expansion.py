"""
finradar.search.query_expansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

한/영 동의어 확장으로 FTS (PostgreSQL tsquery) 매칭 품질을 향상시킨다.

배경
----
기존 플로우는 ``plainto_tsquery('english', 'user query')`` 를 사용.
이는 사용자 입력의 모든 토큰을 AND로 묶고 특수문자를 안전하게 이스케이프하는
반면, 동의어 매칭을 전혀 지원하지 않는다.

예) 사용자가 "오일 가격 상승"을 입력해도 DB 기사들은 "유가", "원유"를 사용
    → FTS 완전 실패 → cosine-only 결과에 의존

해결
----
토큰 단위로 :py:data:`SYNONYMS` 딕셔너리를 조회해 동의어 그룹을 ``(t1 | t2 | t3)``
형태로 확장한 후, ``to_tsquery`` 로 전달한다. 원 토큰은 반드시 포함되며, 동의어가
없는 토큰은 그대로 AND 체인에 남는다.

Usage
-----
    >>> q = expand_query("오일 가격 상승")
    >>> q.tsquery_expr
    "( '오일' | '유가' | '원유' | '석유' | 'oil' ) & '가격' & '상승'"
    >>> q.expanded_tokens
    {'오일': ['유가', '원유', '석유', 'oil'], ...}
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Synonym dictionary  — 금융/경제 도메인 한영 동의어
# ---------------------------------------------------------------------------
#
# 규칙:
#   - 키는 '자주 쓰이는 표현' 기준, 값은 키 포함 전체 그룹(자기 자신 포함)
#   - 양방향 일관성 권장: "오일" -> [..., "유가", ...] 이면 "유가" -> [..., "오일", ...]
#   - 대소문자 구분 없음 (lookup 시 lower()로 정규화)

SYNONYMS: dict[str, list[str]] = {
    # --- 유가/원유/에너지 ----------------------------------------------------
    "오일": ["오일", "유가", "원유", "석유", "oil"],
    "유가": ["유가", "오일", "원유", "석유", "oil"],
    "원유": ["원유", "유가", "오일", "석유", "crude"],
    "석유": ["석유", "유가", "원유", "오일", "oil"],
    "oil": ["oil", "유가", "원유", "오일", "crude"],
    "brent": ["brent", "브렌트", "유가"],
    "wti": ["wti", "서부텍사스산", "유가"],

    # --- 중앙은행/통화정책 --------------------------------------------------
    # (공백 포함 구문은 tsquery 단일 lexeme이 아니므로 제외)
    "연준": ["연준", "fed", "federal", "미연준"],
    "fed": ["fed", "연준", "federal", "미연준"],
    "한은": ["한은", "한국은행", "bok"],
    "한국은행": ["한국은행", "한은", "bok"],
    "ecb": ["ecb", "유럽중앙은행"],
    "유럽중앙은행": ["유럽중앙은행", "ecb"],
    "boj": ["boj", "일본은행"],
    "일본은행": ["일본은행", "boj"],

    # --- 금리 --------------------------------------------------------------
    "금리": ["금리", "기준금리", "rate", "interest"],
    "기준금리": ["기준금리", "금리", "rate"],
    "rate": ["rate", "금리", "기준금리"],

    # --- 물가/인플레이션 ---------------------------------------------------
    "물가": ["물가", "인플레이션", "inflation", "cpi"],
    "인플레이션": ["인플레이션", "물가", "inflation", "cpi"],
    "inflation": ["inflation", "인플레이션", "물가"],
    "cpi": ["cpi", "소비자물가", "인플레이션"],

    # --- 주식/증시 ---------------------------------------------------------
    "주가": ["주가", "주식", "stock"],
    "증시": ["증시", "주식시장"],
    "코스피": ["코스피", "kospi", "코스피지수"],
    "kospi": ["kospi", "코스피"],
    "코스닥": ["코스닥", "kosdaq"],
    "kosdaq": ["kosdaq", "코스닥"],
    "나스닥": ["나스닥", "nasdaq"],
    "nasdaq": ["nasdaq", "나스닥"],
    "다우": ["다우", "dow", "다우존스"],
    "dow": ["dow", "다우"],
    "s&p": ["s&p", "sp500", "에스앤피"],

    # --- 환율/통화 ---------------------------------------------------------
    "환율": ["환율", "fx", "외환"],
    "원화": ["원화", "krw", "won"],
    "달러": ["달러", "usd", "dollar"],
    "dollar": ["dollar", "달러", "usd"],
    "엔화": ["엔화", "jpy", "yen"],
    "유로": ["유로", "eur", "euro"],

    # --- 한국 대기업 -------------------------------------------------------
    "삼성전자": ["삼성전자", "삼전", "samsung", "005930"],
    "삼전": ["삼전", "삼성전자", "samsung"],
    "samsung": ["samsung", "삼성전자", "삼성"],
    "sk하이닉스": ["sk하이닉스", "하이닉스", "hynix"],
    "하이닉스": ["하이닉스", "sk하이닉스", "hynix"],
    "hynix": ["hynix", "sk하이닉스", "하이닉스"],
    "현대차": ["현대차", "현대자동차", "hyundai"],
    "lg에너지": ["lg에너지", "엘지에너지"],

    # --- 미국 대기업 -------------------------------------------------------
    "엔비디아": ["엔비디아", "nvidia", "nvda"],
    "nvidia": ["nvidia", "엔비디아", "nvda"],
    "nvda": ["nvda", "nvidia", "엔비디아"],
    "애플": ["애플", "apple", "aapl"],
    "apple": ["apple", "애플", "aapl"],
    "aapl": ["aapl", "apple", "애플"],
    "테슬라": ["테슬라", "tesla", "tsla"],
    "tesla": ["tesla", "테슬라", "tsla"],
    "tsla": ["tsla", "tesla", "테슬라"],
    "구글": ["구글", "google", "googl", "alphabet"],
    "google": ["google", "구글", "alphabet", "googl"],
    "마소": ["마소", "마이크로소프트", "microsoft", "msft"],
    "마이크로소프트": ["마이크로소프트", "microsoft", "msft", "마소"],
    "microsoft": ["microsoft", "마이크로소프트", "msft"],

    # --- 반도체/AI ---------------------------------------------------------
    "반도체": ["반도체", "semiconductor", "chip", "칩"],
    "semiconductor": ["semiconductor", "반도체", "chip"],
    "hbm": ["hbm", "고대역폭메모리"],
    "ai": ["ai", "인공지능"],
    "인공지능": ["인공지능", "ai"],
    "lm": ["lm", "llm", "대규모언어모델"],
    "llm": ["llm", "대규모언어모델", "lm"],

    # --- 암호화폐 ----------------------------------------------------------
    "비트코인": ["비트코인", "bitcoin", "btc"],
    "bitcoin": ["bitcoin", "비트코인", "btc"],
    "btc": ["btc", "비트코인", "bitcoin"],
    "이더리움": ["이더리움", "ethereum", "eth"],
    "ethereum": ["ethereum", "이더리움", "eth"],
    "크립토": ["크립토", "암호화폐", "crypto", "가상자산"],
    "암호화폐": ["암호화폐", "크립토", "crypto", "가상자산"],

    # --- 감성/시장 표현 ----------------------------------------------------
    "상승": ["상승", "급등", "오름", "rise", "up"],
    "급등": ["급등", "상승", "폭등", "surge"],
    "하락": ["하락", "급락", "내림", "fall", "down", "drop"],
    "급락": ["급락", "폭락", "하락"],
    "폭락": ["폭락", "급락", "crash"],
    "호황": ["호황", "boom"],
    "불황": ["불황", "recession", "경기침체"],
}


# ---------------------------------------------------------------------------
# Expansion result type
# ---------------------------------------------------------------------------


@dataclass
class ExpandedQuery:
    """Result of :py:func:`expand_query`.

    Attributes
    ----------
    original:
        The original user query, stripped of leading/trailing whitespace.
    tsquery_expr:
        A ``to_tsquery``-compatible expression string. If no token matched
        the synonym dictionary, this is still produced (each token quoted
        and ANDed) so the caller can uniformly use ``to_tsquery``.
    expanded_tokens:
        Mapping of input token → its added synonyms (not including itself).
        Empty when no expansion occurred.
    use_to_tsquery:
        Hint for SQL builders: True when expansion added OR groups and the
        caller should switch from ``plainto_tsquery`` to ``to_tsquery``.
        False when the expression is a simple AND of single tokens (either
        is fine, but ``plainto_tsquery`` is safer on malformed input).
    """

    original: str
    tsquery_expr: str
    expanded_tokens: dict[str, list[str]] = field(default_factory=dict)
    use_to_tsquery: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Valid tsquery lexeme characters: letters (incl. CJK), digits, _, -, .
# We reject anything else so the generated expression can't break to_tsquery.
_LEXEME_SAFE = re.compile(r"[\w가-힣\-.]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Extract lexeme-safe tokens from user input (drops operators / quotes)."""
    return _LEXEME_SAFE.findall(text)


def _quote(lexeme: str) -> str:
    """Wrap a lexeme in single quotes for ``to_tsquery``.

    Single quotes inside the lexeme are doubled, but since we filter inputs
    with ``_LEXEME_SAFE`` the only characters we ever see are safe. The
    quoting still protects against anything unexpected.
    """
    escaped = lexeme.replace("'", "''")
    return f"'{escaped}'"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expand_query(text: str) -> ExpandedQuery:
    """Expand a user query into a synonym-aware ``to_tsquery`` expression.

    Args:
        text: The raw user query.

    Returns:
        :py:class:`ExpandedQuery` — always returns, even for empty input
        (``tsquery_expr`` will be an empty string in that case).
    """
    text = (text or "").strip()
    if not text:
        return ExpandedQuery(original="", tsquery_expr="", use_to_tsquery=False)

    tokens = _tokenize(text)
    if not tokens:
        return ExpandedQuery(original=text, tsquery_expr="", use_to_tsquery=False)

    groups: list[str] = []
    expansions: dict[str, list[str]] = {}

    for tok in tokens:
        lookup = tok.lower()
        synonyms = SYNONYMS.get(lookup)

        if not synonyms:
            groups.append(_quote(tok))
            continue

        # Ensure original token is in the group, de-duplicated, preserving order.
        seen: dict[str, None] = {}  # dict to preserve insertion order
        seen[tok.lower()] = None
        for s in synonyms:
            seen.setdefault(s.lower(), None)
        unique = list(seen.keys())

        if len(unique) == 1:
            groups.append(_quote(tok))
        else:
            group_str = "(" + " | ".join(_quote(s) for s in unique) + ")"
            groups.append(group_str)
            # Report additions (exclude original)
            added = [s for s in unique if s != tok.lower()]
            if added:
                expansions[tok] = added

    tsquery_expr = " & ".join(groups)
    return ExpandedQuery(
        original=text,
        tsquery_expr=tsquery_expr,
        expanded_tokens=expansions,
        use_to_tsquery=bool(expansions),
    )
