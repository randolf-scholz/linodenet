r"""Test JIT-compatibility of `linodenet.projections.testing`."""

from collections import defaultdict

import pytest
import torch

from linodenet.regularizations import (
    REGULARIZATION_FNS,
    REGULARIZATION_MODULES,
)

EXTRA_ARGS: defaultdict[str, tuple[tuple, dict]] = defaultdict(
    lambda: ((), {}),
    Banded=((), {"lower": -2, "upper": +1}),
    Contraction=((), {"lipschitz_bound": 0.7}),
    LipschitzBounded=((), {"lipschitz_bound": 1.2}),
    LowRank=((), {"rank": 1}),
    Masked=((), {"mask": torch.randint(0, 2, (4,), dtype=torch.bool)}),
)


@pytest.mark.parametrize("regularization_name", REGULARIZATION_FNS)
def test_jit_compatibility_functional(regularization_name: str) -> None:
    r"""Test JIT-compatibility of functional projections."""
    x = torch.randn(4, 4)
    projection = REGULARIZATION_FNS[regularization_name]
    scripted_projection = torch.jit.script(projection)

    try:
        result_prior = projection(x)
    except NotImplementedError:
        pytest.skip(f"{regularization_name} is not implemented.")

    result_post = scripted_projection(x)
    assert torch.allclose(result_prior, result_post)


@pytest.mark.parametrize("regularization_name", REGULARIZATION_MODULES)
def test_jit_compatibility_modular(regularization_name: str) -> None:
    r"""Test JIT-compatibility of modular projections."""
    x = torch.randn(4, 4)
    projection_type = REGULARIZATION_MODULES[regularization_name]
    extra_args, extra_kwargs = EXTRA_ARGS[regularization_name]
    projection = projection_type(*extra_args, **extra_kwargs)
    scripted_projection = torch.jit.script(projection)

    try:
        result_prior = projection(x)
    except NotImplementedError:
        pytest.skip(f"{regularization_name} is not implemented.")

    result_post = scripted_projection(x)
    assert torch.allclose(result_prior, result_post)


@pytest.mark.skip(reason="Slow.")
@pytest.mark.parametrize("regularization_name", REGULARIZATION_FNS)
def test_compile_compatibility_functional(regularization_name: str) -> None:
    r"""Test JIT-compatibility of functional projections."""
    x = torch.randn(4, 4)
    projection = REGULARIZATION_FNS[regularization_name]
    scripted_projection = torch.compile(projection)

    try:
        result_prior = projection(x)
    except NotImplementedError:
        pytest.skip(f"{regularization_name} is not implemented.")

    result_post = scripted_projection(x)
    assert torch.allclose(result_prior, result_post)


@pytest.mark.skip(reason="Slow.")
@pytest.mark.parametrize("regularization_name", REGULARIZATION_MODULES)
def test_compile_compatibility_modular(regularization_name: str) -> None:
    r"""Test JIT-compatibility of modular projections."""
    x = torch.randn(4, 4)
    projection_type = REGULARIZATION_MODULES[regularization_name]
    projection = projection_type()
    scripted_projection = torch.compile(projection)

    try:
        result_prior = projection(x)
    except NotImplementedError:
        pytest.skip(f"{regularization_name} is not implemented.")

    result_post = scripted_projection(x)
    assert torch.allclose(result_prior, result_post)
