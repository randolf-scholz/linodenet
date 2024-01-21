r"""Embedding Models."""

__all__ = [
    # Constants
    "EMBEDDINGS",
    # ABCs & Protocols
    "Embedding",
    "EmbeddingABC",
    "LinearEmbedding",
    # Classes
    "ConcatEmbedding",
    "ConcatProjection",
]

from linodenet.models.embeddings._embeddings import (
    ConcatEmbedding,
    ConcatProjection,
    Embedding,
    EmbeddingABC,
    LinearEmbedding,
)

EMBEDDINGS: dict[str, type[Embedding]] = {
    "ConcatEmbedding": ConcatEmbedding,
    "ConcatProjection": ConcatProjection,
    "LinearEmbedding": LinearEmbedding,
}
r"""Dictionary of available embeddings."""
