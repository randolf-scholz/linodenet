r"""Embedding Models."""

__all__ = [
    # Types
    "Embedding",
    "EmbeddingABC",
    # Meta-Objects
    "EMBEDDINGS",
    # Classes
    "ConcatEmbedding",
    "ConcatProjection",
    "LinearEmbedding",
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
