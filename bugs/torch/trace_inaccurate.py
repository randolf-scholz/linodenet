#!/usr/bin/env python


import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.autonotebook import tqdm


def main():
    B = 10_000
    N = 512
    reference_trace = []
    torch_trace = []
    manual_trace = []

    X = np.random.randn(B, N, N).astype(np.float32)
    numpy_values = np.trace(X, axis1=-2, axis2=-1)

    for x in tqdm(X):
        y = torch.from_numpy(x)
        d = np.diag(x)
        reference_trace.append(math.fsum(d))
        torch_trace.append(torch.trace(y))
        manual_trace.append(np.sum(d))

    reference_values = np.array(reference_trace)
    torch_values = np.array(torch_trace)
    manual_values = np.array(manual_trace)

    error_numpy = np.abs(reference_values - numpy_values) / reference_values.__abs__()
    error_torch = np.abs(reference_values - torch_trace) / reference_values.__abs__()
    error_manual = np.abs(reference_values - manual_values) / reference_values.__abs__()

    fig, ax = plt.subplots()
    ax.set_xlabel("error")
    ax.set_ylabel("likelihood")
    ax.hist(error_numpy, bins=100, label="numpy", alpha=0.5, density=True, log=True)
    ax.hist(error_torch, bins=100, label="torch", alpha=0.5, density=True, log=True)
    # ax.hist(error_manual, bins=100, label="manual", alpha=0.5, density=True, log=True)
    ax.set_xscale("log")
    ax.legend()


if __name__ == "__main__":
    main()
    plt.show()
