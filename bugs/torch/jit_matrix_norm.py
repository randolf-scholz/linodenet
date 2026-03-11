#!/usr/bin/env python
# FIXME: https://github.com/pytorch/pytorch/issues/111029


import torch
from torch import Tensor, jit


def wrapped_matrix_norm2(x: Tensor, p: str | int = "fro") -> Tensor:
    if isinstance(p, int):  # completely redundant
        return torch.linalg.matrix_norm(x, ord=p)
    return torch.linalg.matrix_norm(x, ord=p)


jit.script(wrapped_matrix_norm2)  # ✔ no error


def wrapped_matrix_norm(x: Tensor, p: str | int = "fro") -> Tensor:
    return torch.linalg.matrix_norm(x, ord=p)


jit.script(wrapped_matrix_norm)  # ✘ error
