r"""Implementation of concatenated ReLU (CReLU) activation function."""

__all__ = ["crelu", "CReLU"]

import torch
from torch import Tensor, nn


def crelu(x: Tensor) -> tuple[Tensor, Tensor]:
    r"""Concatenated ReLU activation function.

    .. math:: ϕ(x) = [relu(x), relu(-x)]

    >>> result = crelu(torch.tensor([-1, 0, 2]))

    References:
        - | Shang, Wenling, Kihyuk Sohn, Diogo Almeida, and Honglak Lee.
          | “Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units.”
          | Proceedings of The 33rd International Conference on Machine Learning 2016
          | https://proceedings.mlr.press/v48/shang16.html.
    """
    return torch.relu(x), torch.relu(-x)


class CReLU(nn.Module):
    r"""Concatenated ReLU activation function.

    References:
        - | Shang, Wenling, Kihyuk Sohn, Diogo Almeida, and Honglak Lee.
          | “Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units.”
          | Proceedings of The 33rd International Conference on Machine Learning 2016
          | https://proceedings.mlr.press/v48/shang16.html.
    """

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return crelu(x)
