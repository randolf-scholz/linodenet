r"""Test whether `LinODEnet` is forward stable."""

import logging

import pytest
import torch

import linodenet.filters.containers
import linodenet.filters.deprecated
from linodenet import embeddings, system
from linodenet.encoders import ResNet
from linodenet.forecasting import LinODEnet


@pytest.mark.skip(reason="Not implemented yet.")
def test_model_stability() -> None:
    r"""TODO: Implement this test."""
    logger = logging.getLogger(f"{__name__}/{LinODEnet.__name__}")
    logger.info("Testing stability.")

    N, D, L = 1000, 5, 32
    MODEL_CONFIG = {
        "__name__": "LinODEnet",
        "input_size": D,
        "hidden_size": L,
        "embedding_type": "concat",
        "Filter": linodenet.filters.deprecated.FilterList.HP,
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
