r"""Embedding components.

An embedding is an injective mapping $f:X → Y$, that is:

1. It is left-invertible, i.e. there exists a mapping $g:Y → X$ such that
   $g(f(x)) = x$ for all $x ∈ X$, but not necessarily $f(g(y)) = y$ for all $y ∈ Y$.
2. The output dimensionality is (generally) larger than the input dimensionality.
3. we require both an `forward` and a `left_inverse` method, aliased to `encode` and `decode`.
"""

__all__ = [
    # Constants
    "EMBEDDINGS",
    # ABCs & Protocols
    "Embedding",
    "EmbeddingBase",
    # Classes
    "ConcatEmbedding",
    "LinearEmbedding",
]

from linodenet.nn.embeddings.embeddings import (
    ConcatEmbedding,
    Embedding,
    EmbeddingBase,
    LinearEmbedding,
)

EMBEDDINGS: dict[str, type[EmbeddingBase]] = {
    "ConcatEmbedding"  : ConcatEmbedding,
    "LinearEmbedding"  : LinearEmbedding,
}  # fmt: skip
r"""Dictionary of available embeddings."""
