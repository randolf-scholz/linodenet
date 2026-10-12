r"""Implementation of concatenated ReLU (CReLU) activation function."""

__all__ = ["crelu", "CReLU"]

import torch
from torch import Tensor, nn

from signatures import signature


@signature("[(..., *ds)] -> [(..., *ds), (..., *ds)]")
def crelu(x: Tensor) -> tuple[Tensor, Tensor]:
    r"""Concatenated ReLU activation function.

    $$ ϕ(x) = (\relu(x), \relu(-x)) $$

    The result is a pair of tensors, each with the input shape. Concatenation
    along a feature dimension is left to the caller.

    Examples:
        ```pycon
        >>> from imtskit.nn.activations.crelu import crelu
        >>> import torch
        >>> z = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        >>> crelu(z)
        (tensor([0., 0., 1., 2.]), tensor([1., -0., 0., 0.]))

        ```

    References:
        - Shang, Wenling, Kihyuk Sohn, Diogo Almeida, and Honglak Lee. \
          “Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units.” \
          Proceedings of The 33rd International Conference on Machine Learning 2016 \
          <https://proceedings.mlr.press/v48/shang16.html>
    """
    return torch.relu(x), torch.relu(-x)


class CReLU(nn.Module):
    r"""Concatenated ReLU activation function.

    Returns the same pair of tensors as `crelu`; it does not concatenate them.

    References:
        - Shang, Wenling, Kihyuk Sohn, Diogo Almeida, and Honglak Lee. \
          “Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units.” \
          Proceedings of The 33rd International Conference on Machine Learning 2016 \
          <https://proceedings.mlr.press/v48/shang16.html>
    """

    @signature("[(..., *ds)] -> [(..., *ds), (..., *ds)]")
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return crelu(x)
