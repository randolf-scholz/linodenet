r"""Test whether `LinODEnet` is forward stable."""

import logging

import pytest
import torch

from linodenet.forecasting import LinODEnet
from linodenet.mappings import embeddings
from linodenet.nn import ResNet
from linodenet.state_propagation import LinearFlow
from linodenet.state_update import deprecated


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
        "Filter": deprecated.UpdateList,
        "System": {
            "__module__": LinearFlow.__module__,
            "__name__": LinearFlow.__qualname__,
            "kernel_initialization": "skew-symmetric",
        },
        "Encoder": ResNet,
        "Decoder": ResNet,
        "Embedding": embeddings.ConcatEmbedding,
    }
    print(MODEL_CONFIG)
    T = torch.randn(N)
    X = torch.randn(N, D)
    model = LinODEnet(D, L)
    model(T, X)
