r"""Embedding Models."""

__all__ = [
    # Types
    "Embedding",
    # Meta-Objects
    "EMBEDDINGS",
    # Classes
    "ConcatEmbedding",
    "ConcatProjection",
]

import logging
from typing import Final

from torch import nn

from linodenet.models.embeddings._embeddings import ConcatEmbedding, ConcatProjection

__logger__ = logging.getLogger(__name__)

Embedding = nn.Module
r"""Type hint for Embeddings."""


EMBEDDINGS: Final[dict[str, type[Embedding]]] = {
    "ConcatEmbedding": ConcatEmbedding,
    "ConcatProjection": ConcatProjection,
}
r"""Dictionary of available embeddings."""
