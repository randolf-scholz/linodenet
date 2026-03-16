import pytest
import torch
from torch import linspace, tensor

from linodenet_special.compiled import RAW_KERNELS

XFAIL_OPCHECK = {
    "singular_triplet": "AOT opcheck fails on data-dependent control flow in the current kernel.",
    "spectral_norm": "AOT opcheck fails on data-dependent control flow in the current kernel.",
}

EXAMPLE_ARGS: dict[str, dict[str, tuple[tuple, dict]]] = {
    "bimodal_to_gaussian": {
        "vector": (
            (linspace(-3.0, 3.0, steps=9), tensor(2.0), tensor(1.0)),
            {},
        ),
    },
    "gaussian_to_bimodal": {
        "vector": (
            (linspace(-2.0, 2.0, steps=9), tensor(2.0), tensor(1.0)),
            {},
        ),
    },
    "gaussian_to_mixture": {
        "vector": (
            (
                linspace(-2.0, 2.0, steps=7),
                tensor([0.2, 0.3, 0.5]),
                tensor([-2.0, 0.0, 1.5]),
                tensor([0.8, 1.0, 1.2]),
            ),
            {},
        ),
    },
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
    "mixture_to_gaussian": {
        "vector": (
            (
                linspace(-3.0, 3.0, steps=7),
                tensor([0.2, 0.3, 0.5]),
                tensor([-2.0, 0.0, 1.5]),
                tensor([0.8, 1.0, 1.2]),
            ),
            {},
        ),
    },
    "ndtri_exp": {
        "scalar": ((-torch.rand(()),), {}),
        "vector": ((-torch.rand(10),), {}),
        "batch": ((-torch.rand(4, 3),), {}),
    },
    "singular_triplet": {
        "matrix": (
            (torch.randn(4, 3),),
            {
                "u0": torch.randn(4),
                "v0": torch.randn(3),
                "maxiter": 8,
                "atol": 1e-6,
                "rtol": 1e-6,
            },
        ),
    },
    "spectral_norm": {
        "matrix": (
            (torch.randn(4, 3),),
            {
                "u0": torch.randn(4),
                "v0": torch.randn(3),
                "maxiter": 8,
                "atol": 1e-6,
                "rtol": 1e-6,
            },
        ),
    },
}


@pytest.mark.parametrize("name", RAW_KERNELS)
def test_opcheck(name: str) -> None:
    if (impl := RAW_KERNELS.get(name)) is None:
        pytest.skip(f"No implementation for {name}", allow_module_level=True)
    if (cases := EXAMPLE_ARGS.get(name)) is None:
        pytest.xfail()
    if (reason := XFAIL_OPCHECK.get(name)) is not None:
        pytest.xfail(reason)

    for case, (args, kwargs) in cases.items():
        torch.library.opcheck(impl, args, kwargs)  # type: ignore[arg-type]
        print(f"{case}: pass")
