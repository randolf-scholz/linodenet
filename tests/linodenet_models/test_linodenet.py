r"""Tests for LinODEnet model construction helpers."""

from types import MappingProxyType

import torch
from torch import nn

from linodenet_models.linodenet import LinearFlow, LinODEnet, make_linodenet
from linodenet_models.state_update import GradientStepUpdater, LpLoss


def test_make_linodenet_instantiates_expected_components() -> None:
    r"""The helper should assemble the default LinearFlow/Linear/update stack."""
    model = make_linodenet(
        linodenet=MappingProxyType(
            {"input_size": 3, "latent_size": 5, "batch_first": False}
        ),
        decoder=MappingProxyType({"in_features": 5, "out_features": 3, "bias": False}),
        state_propagator=MappingProxyType(
            {
                "input_size": 5,
                "kernel_initialization": "zero",
                "kernel_parametrization": "identity",
                "use_rezero": False,
                "use_bias": True,
            }
        ),
        state_updater=MappingProxyType(
            {
                "loss": "l1",
                "regularizer": "l2",
                "regularization_strength": 0.25,
                "step_size": 0.5,
            }
        ),
    )

    assert isinstance(model, LinODEnet)
    assert isinstance(model.decoder, nn.Linear)
    assert isinstance(model.state_update, GradientStepUpdater)
    assert isinstance(model.state_propagator, LinearFlow)
    assert model.decoder is model.state_update.decoder
    assert model.batch_first is False
    assert model.decoder.in_features == 5
    assert model.decoder.out_features == 3
    assert model.decoder.bias is None
    assert model.state_propagator.input_size == 5
    assert model.state_propagator.bias is not None
    assert model.state_propagator.use_rezero is False
    assert isinstance(model.state_update.loss, LpLoss)
    assert isinstance(model.state_update.regularizer, LpLoss)
    assert model.state_update.loss.p == 1.0
    assert model.state_update.regularizer.p == 2.0
    torch.testing.assert_close(
        model.state_update.regularization_strength.detach(),
        torch.tensor(0.25),
    )
    torch.testing.assert_close(model.state_update.step_size.detach(), torch.tensor(0.5))


def test_make_linodenet_uses_decoder_kwargs_for_prediction_shape() -> None:
    r"""The helper should wire a decoder compatible with the model dimensions."""
    model = make_linodenet(
        linodenet={"input_size": 2, "latent_size": 4},
        decoder={"in_features": 4, "out_features": 2},
        state_propagator={"input_size": 4},
        state_updater={},
    )
    latent_state = torch.randn(7, 4)

    prediction = model.decoder(latent_state)

    assert prediction.shape == (7, 2)
