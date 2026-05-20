r"""Check that the linodenet.parametrizations module is compatible with torch.nn.utils.parametrizations."""

import inspect
from collections.abc import Callable

import pytest
from torch.nn.utils import parametrize as torch_parametrize

import linodenet

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
r"""List of all functions and classes in torch.nn.utils.parametrizations."""


def assert_signatures_compatible(func: Callable, reference: Callable) -> None:
    r"""Assert that functions signature is wider than reference."""
    fun_sig = inspect.signature(func)
    ref_sig = inspect.signature(reference)

    for param in ref_sig.parameters:
        if param not in fun_sig.parameters:
            raise AssertionError(f"Parameter {param} not in function signature!")
        ref_kind = ref_sig.parameters[param].kind
        param_kind = fun_sig.parameters[param].kind
        if param_kind != ref_kind:
            raise AssertionError(
                f"Parameter {param!r} has different kind! (expected {ref_kind}, got {param_kind})"
            )


def test_interface_complete() -> None:
    assert set(torch_parametrize.__all__) == set(TORCH_INTERFACE)


@pytest.mark.parametrize("name", FNS)
def test_signatures_compatibility_torch(name: str) -> None:
    impl = getattr(linodenet.nn.parametrize, name, None)
    ref = getattr(torch_parametrize, name, None)

    if ref is None:
        raise NotImplementedError(
            f"torch.nn.utils.parametrizations.{name} not implemented"
        )
    if impl is None:
        pytest.xfail(f"linodenet.parametrizations.{name} not implemented")

    assert_signatures_compatible(impl, ref)
