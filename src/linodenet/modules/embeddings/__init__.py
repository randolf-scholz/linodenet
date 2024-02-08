r"""Embedding components.

We call layers embedding if they satisfy 3 properties:

1. They provide both a encode and decode method.
2. They are left-invertible, i.e. `decode(encode(x)) = x`, but not necessarily
   `encode(decode(x)) = x`.
3. The output dimensionality is larger than the input dimensionality.
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
    "ConcatEmbedding": ConcatEmbedding,
    "ConcatProjection": ConcatProjection,
    "LinearEmbedding": LinearEmbedding,
}
r"""Dictionary of available embeddings."""
