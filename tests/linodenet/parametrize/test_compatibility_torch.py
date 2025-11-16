r"""Check that the linodenet.parametrize module is compatible with torch.nn.utils.parametrize."""

import pytest
from torch.nn.utils import parametrize as torch_parametrize

import linodenet
from linodenet.testing import assert_signatures_compatible

FNS = [
    "register_parametrization",
    "remove_parametrizations",
    "is_parametrized",
    "cached",
]

CLS = [
    "ParametrizationList",
]

TORCH_INTERFACE = [
    "ParametrizationList",
    "cached",
    "is_parametrized",
    "register_parametrization",
    "remove_parametrizations",
    "transfer_parametrizations_and_params",
    "type_before_parametrizations",
]
r"""List of all functions and classes in torch.nn.utils.parametrize."""


def test_interface_complete() -> None:
    assert set(torch_parametrize.__all__) == set(TORCH_INTERFACE)


@pytest.mark.parametrize("name", FNS)
def test_signatures_compatibility_torch(name: str) -> None:
    impl = getattr(linodenet.parametrize, name, None)
    ref = getattr(torch_parametrize, name, None)

    if ref is None:
        raise NotImplementedError(f"torch.nn.utils.parametrize.{name} not implemented")
    if impl is None:
        pytest.xfail(f"linodenet.parametrize.{name} not implemented")

    assert_signatures_compatible(impl, ref)
