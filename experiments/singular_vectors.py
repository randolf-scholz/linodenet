#!/usr/bin/env python
r"""Plotting singular vectors of random matrices."""

import numpy as np
from matplotlib import pyplot as plt

MAXITER, M, N = 1_000_000, 2, 2
matrices = np.random.randn(MAXITER, M, N)
us = np.full((MAXITER, M), fill_value=float("nan"))
vs = np.full((MAXITER, N), fill_value=float("nan"))

for k, A in enumerate(matrices):
    U, S, Vh = np.linalg.svd(A)
    u, s, v = U[:, 0], S[0], Vh[0, :]
    us[k] = u
    vs[k] = v

angle_u = np.arctan2(*us.T[::-1])  # takes y as first arg
angle_v = np.arctan2(*vs.T[::-1])  # takes y as first arg


fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300, constrained_layout=True)
ax.plot(angle_u, angle_v, ".", ms=2, markeredgecolor="none", alpha=0.2)
ax.set_xlabel("angle u")
ax.set_ylabel("angle v")
ax.set_aspect("equal", adjustable="box")
ax.set_xticks([-np.pi, 0, np.pi], ["-π", "0", "+π"])
ax.set_yticks([-np.pi, 0, np.pi], ["-π", "0", "+π"])
fig.savefig("hexagons.png", dpi=300)
