r"""Test whether `LinODEnet` is forward stable."""

import logging

import pytest
import torch

from linodenet.modules import LinODEnet, ResNet, embeddings, filters, system

__logger__ = logging.getLogger(__name__)


@pytest.mark.skip(reason="Not implemented yet.")
def test_model_stability() -> None:
    """TODO: Implement this test."""
    LOGGER = __logger__.getChild(LinODEnet.__name__)
    LOGGER.info("Testing stability.")

    N, D, L = 1000, 5, 32
    MODEL_CONFIG = {
        "__name__": "LinODEnet",
        "input_size": D,
        "hidden_size": L,
        "embedding_type": "concat",
        "Filter": filters.base.SequentialFilter.HP,
        "System": system.LinODECell.HP | {"kernel_initialization": "skew-symmetric"},
        "Encoder": ResNet.HP,
        "Decoder": ResNet.HP,
        "Embedding": embeddings.ConcatEmbedding.HP,
    }
    print(MODEL_CONFIG)
    T = torch.randn(N)
    X = torch.randn(N, D)
    model = LinODEnet(D, L)
    model(T, X)
