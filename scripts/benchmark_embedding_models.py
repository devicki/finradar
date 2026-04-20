"""임베딩 모델 품질 비교 벤치마크.

현재 프로덕션 모델(all-MiniLM-L6-v2)과 교체 후보
(paraphrase-multilingual-MiniLM-L12-v2)를 동일 쿼리/문서 쌍에서 비교한다.

평가 지표
---------
1. **관련 vs 무관 cosine 갭** — 같은 쿼리로 '관련 기사'와 '무관 기사'의
   cosine 차이를 측정. 갭이 클수록 구분력이 좋음.

2. **Cross-lingual 매칭** — 한국어 쿼리로 영어 제목 cosine 측정.
   "연준 금리 인상" vs "Fed rate hike" 가 가까워야 좋음.

3. **품질 분포** — 10건 후보 중 정답이 Top-K에 들어가는지.

Usage:
    docker compose exec celery_worker python /app/scripts/benchmark_embedding_models.py
    (celery_worker에서 돌리는 이유: GPU 필요)
"""
from __future__ import annotations

import time

import numpy as np
from sentence_transformers import SentenceTransformer


_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",                         # 레거시 (참고용)
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",   # 현재 프로덕션
    "BAAI/bge-m3",                                                   # 고품질 후보
]


# ---------------------------------------------------------------------------
# 테스트 케이스
# ---------------------------------------------------------------------------
#
# 각 쿼리에 대해 '관련' 문서 3개와 '무관' 문서 3개를 준비.
# 좋은 모델이라면 관련 평균 cosine > 무관 평균 cosine (큰 갭).

_CASES = [
    {
        "query": "연준 금리 인상",
        "query_lang": "ko",
        "relevant": [
            ("ko", "한국은행 기준금리 0.25%p 인상, 물가 안정 최우선"),
            ("ko", "미 연준, 9월 FOMC에서 금리 동결… 인플레이션 경계"),
            ("en", "Fed raises interest rates by 25 basis points amid inflation"),
        ],
        "unrelated": [
            ("ko", "편의점업계, 고유가 피해지원금 앞두고 일제히 할인 행사"),
            ("ko", "빈 사무실에 딸기가 주렁주렁, 공실 늘더니 무슨 일이"),
            ("ko", "'예수 행세 논란' 트럼프, 성경 낭독 행사 참가"),
        ],
    },
    {
        "query": "Nvidia AI chip earnings",
        "query_lang": "en",
        "relevant": [
            ("en", "Nvidia reports record Q3 revenue on AI chip demand"),
            ("en", "NVDA earnings beat estimates as data center sales soar"),
            ("ko", "엔비디아, AI 반도체 수요 폭증으로 3분기 실적 최고치"),
        ],
        "unrelated": [
            ("en", "Starbucks expands pumpkin spice lineup for autumn season"),
            ("ko", "서울 강남 재건축 단지 분양가 상승세"),
            ("en", "European soccer clubs face transfer window restrictions"),
        ],
    },
    {
        "query": "삼성전자 반도체 실적",
        "query_lang": "ko",
        "relevant": [
            ("ko", "삼성전자, HBM 메모리 독주로 3분기 영업이익 10조원 돌파"),
            ("ko", "삼전 반도체 부문 흑자 전환… DRAM·NAND 가격 반등"),
            ("en", "Samsung Electronics Q3 profit surges on semiconductor rebound"),
        ],
        "unrelated": [
            ("ko", "포스코, 친환경 수소환원제철 시험 가동 돌입"),
            ("en", "Apple unveils new MacBook Pro with M4 chip"),
            ("ko", "국내 관광객 감소… 호텔업계 객실 점유율 하락"),
        ],
    },
]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """벡터 하나짜리 cosine (이미 L2-normalised 된 값 가정)."""
    return float(np.dot(a, b))


def _evaluate_model(model_name: str) -> dict:
    print(f"\n{'='*70}\n모델: {model_name}\n{'='*70}")

    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, device="cuda")
    load_sec = time.perf_counter() - t0
    print(f"모델 로드: {load_sec:.1f}초 (embedding dim={model.get_sentence_embedding_dimension()})\n")

    all_gaps: list[float] = []
    all_cross_sims: list[float] = []

    for idx, case in enumerate(_CASES, start=1):
        q_emb = model.encode(case["query"], normalize_embeddings=True)

        rel_sims: list[tuple[str, float, str]] = []
        for lang, text in case["relevant"]:
            emb = model.encode(text, normalize_embeddings=True)
            sim = _cos(q_emb, emb)
            rel_sims.append((lang, sim, text))

        unrel_sims: list[tuple[str, float, str]] = []
        for lang, text in case["unrelated"]:
            emb = model.encode(text, normalize_embeddings=True)
            sim = _cos(q_emb, emb)
            unrel_sims.append((lang, sim, text))

        rel_mean = np.mean([s for _, s, _ in rel_sims])
        unrel_mean = np.mean([s for _, s, _ in unrel_sims])
        gap = rel_mean - unrel_mean
        all_gaps.append(gap)

        # Cross-lingual: 쿼리와 다른 언어인 relevant 기사
        for lang, sim, _ in rel_sims:
            if lang != case["query_lang"]:
                all_cross_sims.append(sim)

        print(f"[Case {idx}] 쿼리: '{case['query']}'  (lang={case['query_lang']})")
        print(f"  관련 기사 cosine:")
        for lang, sim, text in sorted(rel_sims, key=lambda x: -x[1]):
            print(f"    [{lang}] {sim:.3f}  {text[:60]}")
        print(f"  무관 기사 cosine:")
        for lang, sim, text in sorted(unrel_sims, key=lambda x: -x[1]):
            print(f"    [{lang}] {sim:.3f}  {text[:60]}")
        print(f"  → 관련 평균={rel_mean:.3f}, 무관 평균={unrel_mean:.3f}, 갭={gap:.3f}\n")

    print(f"{'-'*70}")
    print(f"요약")
    print(f"{'-'*70}")
    print(f"  평균 갭 (관련 - 무관):    {np.mean(all_gaps):.3f}")
    print(f"  최소 갭:                  {np.min(all_gaps):.3f}")
    print(f"  Cross-lingual 평균 cos:   {np.mean(all_cross_sims):.3f}  (높을수록 다국어 매칭 OK)")
    return {
        "model": model_name,
        "mean_gap": float(np.mean(all_gaps)),
        "min_gap": float(np.min(all_gaps)),
        "cross_lingual_mean": float(np.mean(all_cross_sims)),
    }


def main() -> None:
    results = []
    for name in _MODELS:
        results.append(_evaluate_model(name))

    print(f"\n{'='*70}\n최종 비교\n{'='*70}")
    print(f"{'모델':<60} | {'평균갭':>6} | {'최소갭':>6} | {'xlingual':>8}")
    print("-" * 95)
    for r in results:
        short = r["model"].split("/")[-1]
        print(
            f"{short:<60} | {r['mean_gap']:>6.3f} | {r['min_gap']:>6.3f} | "
            f"{r['cross_lingual_mean']:>8.3f}"
        )

    # 페어 비교: 바로 이전 모델 대비 개선폭
    print(f"\n{'='*70}\n페어별 개선폭 (이전 모델 대비)\n{'='*70}")
    for i in range(1, len(results)):
        prev = results[i - 1]
        cur = results[i]
        gap_delta = cur["mean_gap"] - prev["mean_gap"]
        cross_delta = cur["cross_lingual_mean"] - prev["cross_lingual_mean"]
        prev_short = prev["model"].split("/")[-1]
        cur_short = cur["model"].split("/")[-1]
        print(f"  {prev_short}  →  {cur_short}")
        print(f"    평균 갭 변화:       {gap_delta:+.3f}")
        print(f"    cross-lingual 변화: {cross_delta:+.3f}")

    # 현재 프로덕션(index 1) 대비 후보(index 2+) 판정
    if len(results) >= 3:
        prod = results[1]
        candidate = results[2]
        gap_improvement = candidate["mean_gap"] - prod["mean_gap"]
        cross_improvement = candidate["cross_lingual_mean"] - prod["cross_lingual_mean"]
        print(f"\n{candidate['model'].split('/')[-1]} 후보 판정")
        print(f"  구분력 개선 (갭):         {gap_improvement:+.3f}")
        print(f"  Cross-lingual 개선:       {cross_improvement:+.3f}")
        if gap_improvement > 0.05 and cross_improvement > 0.05:
            print("\n✅ 교체 추천: 구분력 + 다국어 모두 유의미하게 개선")
        elif gap_improvement > 0.05 or cross_improvement > 0.05:
            print("\n🟡 교체 고려: 한 쪽만 개선")
        else:
            print("\n❌ 교체 비권장: 개선폭 미미 (기존 모델 유지)")


if __name__ == "__main__":
    main()
