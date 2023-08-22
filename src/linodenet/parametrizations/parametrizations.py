"""Parametrizations for Torch."""

__all__ = ["Parametrization"]

from abc import ABC, abstractmethod
from typing import Protocol

import torch
from torch import Tensor, nn


class ParametrizationProto(Protocol):
    """Protocol for parametrizations."""


class ParametrizationABC(nn.Module, ABC):
    """Abstract base class for parametrizations."""

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset the parameters of the parametrization."""
        ...

    @abstractmethod
    def reset_cache(self) -> None:
        """Reset the cached weight matrix."""
        ...

    @abstractmethod
    def recompute_cache(self) -> None:
        """Recompute the cached weight matrix."""
        ...

    @abstractmethod
    def projection(self) -> None:
        """Project the cached weight matrix."""
        ...


class Parametrization(nn.Module):
    """A parametrized tensor."""

    parametrized_tensors: dict[str, Tensor]
    cached_tensors: dict[str, Tensor]

    @torch.no_grad()
    def register_parametrization(self, name: str) -> None:
        """Register a parametrization."""
        if name not in self.named_parameters():
            raise ValueError(f"{name=} is not a known named parameter!")

        # lookup the tensor.
        tensor = getattr(self, name)
        assert isinstance(tensor, nn.Parameter)

        # create the cached tensor.
        self.register_cached_tensor(f"cached_{name}", torch.empty_like(tensor))

        # register the parametrization.
        self.parametrized_tensors[name] = tensor

    @torch.no_grad()
    def register_cached_tensor(self, name: str, tensor: Tensor) -> None:
        """Register a cached tensor."""
        if name in self.cached_tensors:
            raise ValueError(f"Cache with {name=!r} already registered!")
        if name in self.named_buffers():
            raise ValueError(f"Buffer with {name=!r} already taken!")

        self.register_buffer(name, tensor)
        self.cached_tensors[name] = getattr(self, name)

    def recompute_cache(self) -> None:
        # Compute the cached weight matrix
        new_tensors = self.forward()

        # copy the new tensors into the cache
        for key, tensor in new_tensors.items():
            self.cached_tensors[key].copy_(tensor)

    @torch.no_grad()
    def projection(self) -> None:
        # update the cached weight matrix
        self.recompute_cache()

        # copy the cached values into the parametrized tensors
        for key, tensor in self.parametrized_tensors.items():
            tensor.copy_(self.cached_tensors[key])

    def reset_cache(self) -> None:
        # apply projection step.
        self.projection()

        # reengage the autograd engine
        # detach() is necessary to avoid "Trying to backward through the graph a second time" error
        for key, tensor in self.cached_tensors.items():
            tensor.detach_()

        # recompute the cache
        # Note: we need the second run to set up the gradients
        self.recompute_cache()

    def reset_cache_expanded(self) -> None:
        # apply projection step.
        with torch.no_grad():
            # update the cached weight matrix
            new_tensors = self.forward()

            if new_tensors.keys() != self.cached_tensors.keys():
                raise ValueError(
                    f"{new_tensors.keys()=} != {self.cached_tensors.keys()=}"
                )

            # copy the new tensors into the cache
            for key, tensor in new_tensors.items():
                self.cached_tensors[key].copy_(tensor)

        # copy the cached values into the parametrized tensors
        for key, tensor in self.parametrized_tensors.items():
            tensor.copy_(self.cached_tensors[key])

        # reengage the autograd engine
        # detach() is necessary to avoid "Trying to backward through the graph a second time" error
        for key, tensor in self.cached_tensors.items():
            tensor.detach_()

        # recompute the cache
        # Note: we need the second run to set up the gradients
        # Compute the cached weight matrix
        new_tensors = self.forward()

        # copy the new tensors into the cache
        for key, tensor in new_tensors.items():
            self.cached_tensors[key].copy_(tensor)


class LinearContraction(ParametrizationABC):
    def __init__(
        self, input_size: int, output_size: int, L: float = 1.0, *, bias: bool = True
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.L = L

        self.weight = nn.Parameter(torch.empty((output_size, input_size)))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.register_buffer("sigma", torch.tensor(1.0))
        self.register_buffer("u", torch.randn(output_size))
        self.register_buffer("v", torch.randn(input_size))
        self.register_buffer("cached_weight", torch.empty_like(self.weight))
        self.register_buffer("one", torch.ones(1))
        self.register_buffer("c", torch.tensor(self.L))
        self.reset_cache()

    def reset_parameters(self) -> None:
        r"""Reset both weight matrix and bias vector."""
        with torch.no_grad():
            bound: float = float(torch.rsqrt(torch.tensor(self.input_size)))
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def reset_cache(self) -> None:
        """Reset the cached weight matrix.

        Needs to be called after every .backward!
        """
        # apply projection step.
        with torch.no_grad():
            self.recompute_cache()

        self.projection()

        # detach() is necessary to avoid "Trying to backward through the graph a second time" error

        for key, tensor in self.cached_tensors.items():
            tensor.detach_()

        self.sigma.detach_()
        self.u.detach_()
        self.v.detach_()
        self.cached_weight.detach_()

        # recompute the cache
        # NOTE: we need the second run to set up the gradients!
        self.recompute_cache()

    def recompute_cache(self) -> None:
        r"""Recompute the cached weight matrix."""
        new_tensors = self.forward()
        for key, tensor in new_tensors.items():
            self.cached_tensors[key].copy_(tensor)

    @torch.no_grad()
    def projection(self):
        for key, tensor in self.parametrized_tensors.items():
            tensor.copy_(self.cached_tensors[key])

    def forward(self) -> dict[str, Tensor]:
        r""".. Signature:: ``(..., n) -> (..., n)``."""
        sigma, u, v = singular_triplet(self.weight, u0=self.u, v0=self.v)
        gamma = torch.minimum(self.one, self.c / sigma)
        weight = gamma * self.weight
        return {"weight": weight, "sigma": sigma, "u": u, "v": v}
