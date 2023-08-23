"""Parametrizations for Torch."""

__all__ = ["Parametrization"]

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from linodenet.lib import singular_triplet


@runtime_checkable
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
    def register_parametrization(self, name: str, param: nn.Parameter) -> None:
        """Register a parametrization."""
        if not isinstance(param, nn.Parameter):
            raise ValueError(f"Given tensor is not a nn.Parameter!")

        # create the cached tensor.
        self.register_cached_tensor(f"cached_{name}", torch.empty_like(param))

        # register the parametrization.
        self.parametrized_tensors[name] = param

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


class SpectralNormalization(Parametrization):
    """Spectral normalization."""

    # constants
    GAMMA: Tensor
    ONE: Tensor

    # cached
    u: Tensor
    v: Tensor
    sigma: Tensor

    # parametrized
    weight: Tensor

    def __init__(self, weight: nn.Parameter, /, gamma: float = 1.0) -> None:
        super().__init__()

        assert len(weight.shape) == 2
        m, n = weight.shape

        options = {
            "dtype": weight.dtype,
            "device": weight.device,
            "layout": weight.layout,
        }

        # parametrized and cached
        self.register_parametrization("weight", weight)
        self.register_cached_tensor("u", torch.empty(m, **options))
        self.register_cached_tensor("v", torch.empty(n, **options))
        self.register_cached_tensor("sigma", torch.empty(1, **options))

        # constants
        self.register_buffer("ONE", torch.empty(1, **options))
        self.register_buffer("GAMMA", torch.empty(gamma, **options))

    def forward(self) -> dict[str, Tensor]:
        sigma, u, v = singular_triplet(self.weight, u0=self.u, v0=self.v)
        gamma = torch.minimum(self.ONE, self.GAMMA / sigma)
        weight = gamma * self.weight
        return {"weight": weight, "u": u, "v": v, "sigma": sigma}


def reset_all_caches(module: nn.Module) -> None:
    """Reset all caches in a module."""
    for submodule in module.modules():
        if isinstance(submodule, ParametrizationABC):
            submodule.reset_cache()


class reset_caches(AbstractContextManager):
    """reset_caches context manager."""

    def __init__(self, module: nn.Module) -> None:
        self.module = module

    def __enter__(self):
        reset_all_caches(self.module)
        return self.module

    def __exit__(self, exc_type, exc_val, exc_tb):
        reset_all_caches(self.module)
        return False
