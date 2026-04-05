import pytest
import torch
from torch import linspace, tensor

from linodenet_special.compiled import RAW_KERNELS
from tests.testing import pytest_xfail

_bimodal_to_gaussian_cases = {
    "vector": (
        (linspace(-2.0, 2.0, steps=9), tensor(2.0), tensor(1.0)),
        {},
    ),
}

_gaussian_to_bimodal_cases = {
    "vector": (
        (linspace(-2.0, 2.0, steps=9), tensor(2.0), tensor(1.0)),
        {"maxiter": 1},
    ),
}

_mixture_to_gaussian_cases = {
    "vector": (
        (
            linspace(-2.0, 2.0, steps=7),
            tensor([0.2, 0.3, 0.5]),
            tensor([-2.0, 0.0, 1.5]),
            tensor([0.8, 1.0, 1.2]),
        ),
        {},
    ),
}

_gaussian_to_mixture_cases = {
    "vector": (
        (
            linspace(-2.0, 2.0, steps=7),
            tensor([0.2, 0.3, 0.5]),
            tensor([-2.0, 0.0, 1.5]),
            tensor([0.8, 1.0, 1.2]),
        ),
        {"maxiter": 1},
    ),
}

_spectral_norm_cases = {
    "matrix": (
        (torch.randn(4, 3),),
        {
            "u0": torch.randn(4),
            "v0": torch.randn(3),
            "maxiter": 1,
            "atol": 1e-6,
            "rtol": 1e-6,
        },
    ),
}

EXAMPLE_ARGS: dict[str, dict[str, tuple[tuple, dict]]] = {
    "bimodal_to_gaussian": _bimodal_to_gaussian_cases,
    "gaussian_to_bimodal": _gaussian_to_bimodal_cases,
    "gaussian_to_mixture": _gaussian_to_mixture_cases,
    "mixture_to_gaussian": _mixture_to_gaussian_cases,
    "bimodal_to_gaussian_value_and_grad": _bimodal_to_gaussian_cases,
    "gaussian_to_bimodal_value_and_grad": _gaussian_to_bimodal_cases,
    "gaussian_to_mixture_value_and_grad": _gaussian_to_mixture_cases,
    "mixture_to_gaussian_value_and_grad": _mixture_to_gaussian_cases,
    "hard_bend": {
        "scalar": (
            (torch.randn(()), tensor(2.0), tensor(2.0), tensor(1.0)),
            {},
        ),
        "vector": (
            (torch.randn(10), tensor(2.0), tensor(2.0), tensor(1.0)),
            {},
        ),
    },
    "ndtri_exp": {
        "scalar": ((-torch.rand(()),), {}),
        "vector": ((-torch.rand(10),), {}),
        "batch": ((-torch.rand(4, 3),), {}),
    },
    "singular_triplet": _spectral_norm_cases,
    "spectral_norm": _spectral_norm_cases,
}


@pytest.mark.parametrize("name", RAW_KERNELS)
def test_opcheck(name: str) -> None:
    if (impl := RAW_KERNELS.get(name)) is None:
        pytest.skip(f"No implementation for {name}", allow_module_level=True)

    test_cases = EXAMPLE_ARGS[name]

    for case, (args, kwargs) in test_cases.items():
        with pytest_xfail(strict=False):
            torch.library.opcheck(impl, args, kwargs)  # type: ignore[arg-type]
            print(f"{case}: pass")
