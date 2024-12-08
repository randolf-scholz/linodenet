r"""Embedding components.

We call a layer an embedding if they satisfy 3 properties:

1. It has both an `encode` and a `decode` method.
2. It is left-invertible, i.e. `decode(encode(x)) = x`, but not necessarily
   `encode(decode(y)) = y`.
3. The output dimensionality is (generally) larger than the input dimensionality.
"""

__all__ = [
    # Constants
    "EMBEDDINGS",
    # ABCs & Protocols
    "Embedding",
    "EmbeddingABC",
    # Classes
    "ConcatEmbedding",
    "ConcatProjection",
    "LinearEmbedding",
]

from linodenet.modules.embeddings.base import (
    ConcatEmbedding,
    ConcatProjection,
    Embedding,
    EmbeddingABC,
    LinearEmbedding,
)

EMBEDDINGS: dict[str, type[Embedding]] = {
    "ConcatEmbedding"  : ConcatEmbedding,
    "ConcatProjection" : ConcatProjection,
    "LinearEmbedding"  : LinearEmbedding,
}  # fmt: skip
r"""Dictionary of available embeddings."""
