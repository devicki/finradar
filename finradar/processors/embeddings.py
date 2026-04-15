"""
finradar.processors.embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dense vector embedding generation using sentence-transformers.

The default model (all-MiniLM-L6-v2) produces 384-dimensional L2-normalised
vectors that are stored in the ``news_items.embedding`` pgvector column and used
for cosine-similarity search at query time.

Design:
  - Model is loaded lazily on first use to avoid occupying GPU memory during
    startup phases that don't need embeddings.
  - Embeddings are always L2-normalised (``normalize_embeddings=True``) so that
    dot-product and cosine-similarity queries are equivalent in pgvector.
  - ``generate()`` is provided for single-text convenience; production pipelines
    should use ``generate_batch()`` to amortise GPU overhead.

Usage::

    from finradar.processors.embeddings import EmbeddingGenerator

    gen = EmbeddingGenerator()
    vec = gen.generate("Federal Reserve raises rates by 50bp")
    # list of 384 floats

    vecs = gen.generate_batch(["headline 1", "headline 2", ...])
    # list of lists, one per input text
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from finradar.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Generate sentence embeddings for news articles (local GPU).

    Args:
        model_name: HuggingFace / sentence-transformers model identifier.
                    Defaults to ``sentence-transformers/all-MiniLM-L6-v2``.
        device:     PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dimension = 384  # all-MiniLM-L6-v2 output dimension
        self._model: SentenceTransformer | None = None
        self.logger = logging.getLogger("finradar.processors.embeddings")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the sentence-transformers model on first use."""
        if self._model is not None:
            return

        import torch
        from sentence_transformers import SentenceTransformer

        self.logger.info(
            "Loading embedding model %s on %s ...", self.model_name, self.device
        )

        # Fall back to CPU if CUDA was requested but isn't available.
        effective_device = self.device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning(
                "CUDA requested but not available — falling back to CPU for %s",
                self.model_name,
            )
            effective_device = "cpu"
            self.device = "cpu"

        self._model = SentenceTransformer(self.model_name, device=effective_device)
        self.logger.info(
            "Embedding model %s loaded successfully on %s", self.model_name, self.device
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        The returned vector is L2-normalised to unit length so that pgvector
        cosine-similarity queries give correct results.

        Args:
            text: The text to embed (headline, summary, or combined field).

        Returns:
            List of 384 floats representing the sentence embedding.

        Raises:
            RuntimeError: If embedding inference fails unexpectedly.
        """
        self._load_model()

        if not text or not text.strip():
            self.logger.debug(
                "Empty text passed to generate(); returning zero vector."
            )
            return [0.0] * self.dimension

        try:
            embedding = self._model.encode(  # type: ignore[union-attr]
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embedding.tolist()

        except Exception as exc:
            self.logger.error(
                "Embedding generation failed for text (first 120 chars): %r — %s",
                text[:120],
                exc,
                exc_info=True,
            )
            raise RuntimeError(f"Embedding inference failed: {exc}") from exc

    def generate_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> list[list[float]]:
        """Batch embedding generation for throughput-efficient GPU processing.

        Empty and whitespace-only strings receive a zero vector so that the
        returned list always has the same length as *texts* and indices
        stay aligned with the caller's article list.

        Args:
            texts:      List of texts to embed.
            batch_size: Number of texts per GPU forward pass.  The default of
                        64 works well for the RTX 3080 Ti with all-MiniLM-L6-v2.
                        Reduce if you see CUDA out-of-memory errors.

        Returns:
            List of lists, one 384-float vector per input text, in the same
            order as *texts*.

        Raises:
            RuntimeError: If embedding inference fails for a batch.
        """
        self._load_model()

        if not texts:
            return []

        # Identify empty texts so we can substitute zero vectors after encoding.
        empty_mask: list[bool] = []
        safe_texts: list[str] = []
        for t in texts:
            if not t or not t.strip():
                empty_mask.append(True)
                safe_texts.append(".")  # minimal placeholder — will be overwritten
            else:
                empty_mask.append(False)
                safe_texts.append(t)

        try:
            embeddings = self._model.encode(  # type: ignore[union-attr]
                safe_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            result: list[list[float]] = []
            zero_vec = [0.0] * self.dimension
            for i, vec in enumerate(embeddings):
                if empty_mask[i]:
                    result.append(zero_vec)
                else:
                    result.append(vec.tolist())

            return result

        except Exception as exc:
            self.logger.error(
                "Batch embedding failed for %d texts: %s", len(texts), exc, exc_info=True
            )
            raise RuntimeError(
                f"Batch embedding inference failed ({len(texts)} texts): {exc}"
            ) from exc

    @classmethod
    def from_settings(cls) -> "EmbeddingGenerator":
        """Construct an EmbeddingGenerator from application settings.

        Reads ``embedding_model`` and ``local_model_device`` from the
        :py:func:`finradar.config.get_settings` singleton.
        """
        settings = get_settings()
        return cls(
            model_name=settings.embedding_model,
            device=settings.local_model_device,
        )
