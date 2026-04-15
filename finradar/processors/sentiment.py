"""
finradar.processors.sentiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Financial sentiment analysis using FinBERT (English) and KR-FinBERT-SC (Korean).
Both models run locally on GPU (RTX 3080 Ti) for high-throughput batch processing.

Design:
  - Models are loaded lazily on first use to avoid consuming VRAM at startup.
  - ``analyze()`` handles a single text; ``analyze_batch()`` uses efficient batched
    GPU inference and should be preferred for bulk processing.
  - The returned score is ``positive_prob - negative_prob`` in [-1.0, 1.0].
  - Inputs longer than the model's maximum sequence length (512 tokens for both
    models) are automatically truncated by the tokenizer.

Usage::

    from finradar.processors.sentiment import get_sentiment_analyzer

    analyzer = get_sentiment_analyzer("en")
    result = analyzer.analyze("Apple shares surged after record earnings beat.")
    # {"score": 0.87, "label": "positive", "confidence": 0.91}

    results = analyzer.analyze_batch([
        "Fed cuts rates by 25bp, markets rally",
        "Recession fears mount as PMI falls to 46",
    ])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from finradar.config import get_settings

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentAnalyzer:
    """Financial sentiment analysis using ProsusAI/finbert (local GPU).

    FinBERT was trained on financial news and reports.  Its three output classes
    are ordered **[positive, negative, neutral]** (indices 0, 1, 2).

    The continuous sentiment *score* is defined as::

        score = P(positive) - P(negative)   in [-1.0, 1.0]

    The *label* is the argmax class.
    The *confidence* is the probability of the predicted class.
    """

    # FinBERT class order: 0=positive, 1=negative, 2=neutral
    _LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}
    _POSITIVE_IDX = 0
    _NEGATIVE_IDX = 1

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: AutoModelForSequenceClassification | None = None
        self._tokenizer: AutoTokenizer | None = None
        self.logger = logging.getLogger("finradar.processors.sentiment")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the HuggingFace model and tokenizer on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.logger.info("Loading sentiment model %s on %s ...", self.model_name, self.device)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # If CUDA was requested but is unavailable, fall back to CPU gracefully.
        if self.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning(
                "CUDA requested but not available — falling back to CPU for %s",
                self.model_name,
            )
            self.device = "cpu"

        self._model.to(self.device)
        self._model.eval()
        self.logger.info(
            "Sentiment model %s loaded successfully on %s", self.model_name, self.device
        )

    def _logits_to_result(self, logits: "torch.Tensor") -> dict:
        """Convert a single-row logit tensor to a sentiment result dict.

        Args:
            logits: 1-D tensor of shape (num_labels,).

        Returns:
            dict with keys ``score`` (float), ``label`` (str), ``confidence`` (float).
        """
        import torch

        probs = torch.softmax(logits, dim=-1).cpu().float()
        probs_list = probs.tolist()

        label_idx = int(torch.argmax(probs).item())
        label = self._LABEL_MAP[label_idx]
        confidence = float(probs_list[label_idx])

        # Continuous score: P(positive) - P(negative)
        pos_prob = float(probs_list[self._POSITIVE_IDX])
        neg_prob = float(probs_list[self._NEGATIVE_IDX])
        score = round(pos_prob - neg_prob, 6)

        return {
            "score": score,
            "label": label,
            "confidence": round(confidence, 6),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> dict:
        """Analyze the sentiment of a single text.

        Args:
            text: The financial news headline or body to analyze.

        Returns:
            dict::

                {
                    "score":      float,  # -1.0 (most negative) to +1.0 (most positive)
                    "label":      str,    # "positive" | "negative" | "neutral"
                    "confidence": float,  # probability of the predicted class
                }

        Raises:
            RuntimeError: If model inference fails unexpectedly.
        """
        self._load_model()

        if not text or not text.strip():
            self.logger.debug("Empty text passed to analyze(); returning neutral default.")
            return {"score": 0.0, "label": "neutral", "confidence": 1.0}

        import torch

        try:
            inputs = self._tokenizer(  # type: ignore[operator]
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)  # type: ignore[operator]

            return self._logits_to_result(outputs.logits[0])

        except Exception as exc:
            self.logger.error(
                "Sentiment analysis failed for text (first 120 chars): %r — %s",
                text[:120],
                exc,
                exc_info=True,
            )
            raise RuntimeError(f"Sentiment inference failed: {exc}") from exc

    def analyze_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Batch sentiment analysis for throughput-efficient GPU processing.

        Args:
            texts:      List of texts to analyze.
            batch_size: Number of texts to process in each GPU forward pass.
                        Reduce if VRAM is insufficient (RTX 3080 Ti default: 32).

        Returns:
            List of dicts in the same order as *texts*, each with the same
            structure as :py:meth:`analyze`.
        """
        self._load_model()

        if not texts:
            return []

        import torch

        results: list[dict] = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]

            # Replace empty strings with a placeholder to avoid tokenizer errors.
            safe_batch = [t if t and t.strip() else "[PAD]" for t in batch]

            try:
                inputs = self._tokenizer(  # type: ignore[operator]
                    safe_batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._model(**inputs)  # type: ignore[operator]

                for i, logits in enumerate(outputs.logits):
                    original = batch[i]
                    if not original or not original.strip():
                        results.append({"score": 0.0, "label": "neutral", "confidence": 1.0})
                    else:
                        results.append(self._logits_to_result(logits))

            except Exception as exc:
                self.logger.error(
                    "Batch sentiment analysis failed for batch starting at index %d: %s",
                    batch_start,
                    exc,
                    exc_info=True,
                )
                # Append neutral fallbacks for the failed batch so indices stay aligned.
                for _ in batch:
                    results.append({"score": 0.0, "label": "neutral", "confidence": 0.0})

        return results


class KoreanSentimentAnalyzer(SentimentAnalyzer):
    """Korean financial sentiment analysis using snunlp/KR-FinBert-SC.

    KR-FinBERT-SC label order: **0=negative, 1=neutral, 2=positive**.
    This differs from ProsusAI/finbert, so the class constants are overridden.
    """

    # KR-FinBERT-SC class order: 0=negative, 1=neutral, 2=positive
    _LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
    _POSITIVE_IDX = 2
    _NEGATIVE_IDX = 0

    def __init__(
        self,
        model_name: str = "snunlp/KR-FinBert-SC",
        device: str = "cuda",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.logger = logging.getLogger("finradar.processors.sentiment.ko")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_sentiment_analyzer(language: str = "en") -> SentimentAnalyzer:
    """Return the appropriate sentiment analyzer for *language*.

    Args:
        language: ISO 639-1 code.  ``"ko"`` returns a
                  :py:class:`KoreanSentimentAnalyzer`; everything else returns
                  an English :py:class:`SentimentAnalyzer`.

    Returns:
        A :py:class:`SentimentAnalyzer` (or subclass) configured with the
        model and device from application settings.
    """
    settings = get_settings()

    if language == "ko":
        return KoreanSentimentAnalyzer(
            model_name=settings.sentiment_model_ko,
            device=settings.local_model_device,
        )

    return SentimentAnalyzer(
        model_name=settings.sentiment_model_en,
        device=settings.local_model_device,
    )
