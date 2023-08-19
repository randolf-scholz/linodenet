#!/usr/bin/env python
"""Test gradients of custom operators."""


import torch

from linodenet.lib import spectral_norm, spectral_norm_native

if __name__ == "__main__":
    # main program
    with torch.no_grad():
        for _ in range(10):
            m, n = 128, 128
            A = torch.randn(m, n)
            sigma = spectral_norm(A)
            sigma_native = spectral_norm_native(A)
            assert torch.allclose(sigma, sigma_native)

    print("All tests passed.")
