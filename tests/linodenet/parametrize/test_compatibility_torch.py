r"""Check that the linodenet.parametrize module is compatible with torch.nn.utils.parametrize."""

import pytest
from torch.nn.utils import parametrize as torch_parametrize

import linodenet
from linodenet.testing import assert_compatible_signature

FNS = [
    "register_parametrization",
    "remove_parametrizations",
    "is_parametrized",
    "cached",
]

CLS = [
    "ParametrizationList",
]


@pytest.mark.parametrize("name", FNS)
def test_compatibility_torch(name: str) -> None:
    impl = getattr(linodenet.parametrize, name, None)
    ref = getattr(torch_parametrize, name, None)

    if ref is None:
        raise NotImplementedError(f"torch.nn.utils.parametrize.{name} not implemented")
    if impl is None:
        pytest.xfail(f"linodenet.parametrize.{name} not implemented")

    assert_compatible_signature(impl, ref)
