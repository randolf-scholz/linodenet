r"""Regression tests for removing the identity projection."""

import torch

from linodenet.flows.linear import LinearFlow
from linodenet.mappings import MATRIX_PROJECTION_FNS, MATRIX_PROJECTIONS
from linodenet.parametrizations import MATRIX_PARAMETRIZATIONS


def test_identity_projection_is_not_registered() -> None:
    assert "identity" not in MATRIX_PROJECTION_FNS
    assert "Identity" not in MATRIX_PROJECTIONS
    assert "Identity" not in MATRIX_PARAMETRIZATIONS


def test_linear_flow_default_kernel_parametrization_remains_noop() -> None:
    flow = LinearFlow(4)
    weight = torch.randn(4, 4)

    assert torch.equal(flow.kernel_parametrization(weight), weight)
