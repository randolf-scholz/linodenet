#!/usr/bin/env python
"""Test the spectral norm implementation.""" ""

import torch
import torch.utils.cpp_extension

import linodenet


def test_spectral_norm() -> None:
    """Test the spectral norm implementation."""
    source = linodenet.config.PROJECT.ROOT_PATH / "lib" / "spectral_norm.cpp"

    torch.utils.cpp_extension.load(
        name="spectral_norm",
        sources=[source],
        is_python_module=False,
        verbose=True,
    )
    print(torch.ops.custom.spectral_norm)
    spectral_norm = torch.ops.custom.spectral_norm
    A = torch.randn(5, 7)
    s = spectral_norm(A, maxiter=10_000, atol=10**-2, rtol=10**-2)
    print(s)

    if torch.cuda.is_available():
        A = A.cuda()
        s = spectral_norm(A, maxiter=10_000, atol=10**-2, rtol=10**-2)
        print(s)
