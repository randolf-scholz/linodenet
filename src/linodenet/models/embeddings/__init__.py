r"""Embedding Models."""

__all__ = [
    # Types
    "Embedding",
    # Meta-Objects
    "EMBEDDINGS",
    # Classes
    "ConcatEmbedding",
    "ConcatProjection",
    "LinearEmbedding",
]

from typing import Final, TypeAlias

from torch import nn

from linodenet.models.embeddings._embeddings import (
    ConcatEmbedding,
    ConcatProjection,
    LinearEmbedding,
)

Embedding: TypeAlias = nn.Module
r"""Type hint for Embeddings."""


EMBEDDINGS: Final[dict[str, type[nn.Module]]] = {
    "ConcatEmbedding": ConcatEmbedding,
    "ConcatProjection": ConcatProjection,
    "LinearEmbedding": LinearEmbedding,
}
r"""Dictionary of available embeddings."""

del Final, TypeAlias, nn
