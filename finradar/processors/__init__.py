"""
finradar.processors
~~~~~~~~~~~~~~~~~~~

AI processing layer for FinRadar.

Exports:
    SentimentAnalyzer          — English FinBERT-based sentiment analysis (local GPU)
    KoreanSentimentAnalyzer    — Korean KR-FinBERT-SC sentiment analysis (local GPU)
    get_sentiment_analyzer     — Factory that returns the right analyzer for a language
    EmbeddingGenerator         — sentence-transformers embeddings (local GPU)
    LLMProcessor               — Cloud LLM for summarization, translation, metadata extraction
"""

from finradar.processors.embeddings import EmbeddingGenerator
from finradar.processors.llm_processor import LLMProcessor
from finradar.processors.sentiment import (
    KoreanSentimentAnalyzer,
    SentimentAnalyzer,
    get_sentiment_analyzer,
)

__all__ = [
    "SentimentAnalyzer",
    "KoreanSentimentAnalyzer",
    "get_sentiment_analyzer",
    "EmbeddingGenerator",
    "LLMProcessor",
]
