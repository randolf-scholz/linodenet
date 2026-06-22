r"""Test whether compile/export supports distribution outputs."""

import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal
from torch.utils import _pytree


class Demo(nn.Module):
    r"""Return a Gaussian distribution $N(μ(x), 𝕀)$."""

    def __init__(self, dim: int = 3, /) -> None:
        super().__init__()
        self.mu = nn.Linear(dim, dim)
        self.register_buffer("identity", torch.eye(dim))
        self.identity: Tensor

    def forward(self, x: Tensor, /) -> MultivariateNormal:
        return MultivariateNormal(
            self.mu(x),
            covariance_matrix=self.identity,
            validate_args=False,
        )


def test_compile() -> None:
    model = Demo()
    compiled = torch.compile(model, fullgraph=True)
    x = torch.randn(3)

    expected = model(x)
    actual = compiled(x)

    assert torch.allclose(actual.mean, expected.mean)
    assert torch.allclose(actual.covariance_matrix, expected.covariance_matrix)


def test_export() -> None:
    model = Demo()
    x = torch.randn(3)

    # torch.export needs a pytree recipe for distribution outputs.
    _pytree.register_pytree_node(
        MultivariateNormal,
        lambda dist: ([dist.loc, dist.covariance_matrix], None),
        lambda values, _: MultivariateNormal(
            values[0],  # type: ignore[index]
            covariance_matrix=values[1],  # type: ignore[index]
            validate_args=False,
        ),
    )

    exported = torch.export.export(model, args=(x,)).module()
    expected = model(x)
    actual = exported(x)

    assert torch.allclose(actual.mean, expected.mean)
    assert torch.allclose(actual.covariance_matrix, expected.covariance_matrix)
