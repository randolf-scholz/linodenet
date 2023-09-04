"""Parametrizations for Torch.

Idea:

At he end we want to be able to do something as simple as this:

model.weight = MyParametrization(model.weight).weight
"""

__all__ = [
    "ParametrizationProto",
    "ParametrizationABC",
    "Parametrization",
]

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, jit, nn


@runtime_checkable
class ParametrizationProto(Protocol):
    """Protocol for parametrizations.

    Note:
        To work with JIT, the listed methods must be annotated with @jit.export.
    """

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

    @jit.export
    def right_inverse(self) -> None:
        """Compute the right inverse of the parametrization."""
        raise NotImplementedError


class ParametrizationSingleTensor(nn.Module, ParametrizationProto):
    """Parametrization of a single tensor."""

    def __init__(self, parametrization: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.__parametrization = parametrization

    def forward(self) -> Tensor:
        """Apply the parametrization to the weight matrix."""
        return self.parametrization(self.weight)

    @jit.export
    def parametrization(self, x: Tensor) -> Tensor:
        """Apply the parametrization."""
        return self.__parametrization(x)

    @jit.export
    def recompute_cache(self) -> None:
        # Compute the cached weight matrix
        new_tensor = self.forward()
        self.cached_weight.copy_(new_tensor)

    @jit.export
    def projection(self) -> None:
        with torch.no_grad():
            # update the cached weight matrix
            self.recompute_cache()
            self.weight.copy_(self.cached_weight)

    @jit.export
    def reset_cache(self) -> None:
        # apply projection step.
        self.projection()

        # reengage the autograd engine
        # detach() is necessary to avoid "Trying to backward through the graph a second time" error
        self.cached_weight.detach_()

        # recompute the cache
        # Note: we need the second run to set up the gradients
        self.recompute_cache()

    @jit.export
    def reset_cache_expanded(self) -> None:
        with torch.no_grad():
            new_tensor = self.forward()
            self.cached_weight.copy_(new_tensor)
            self.weight.copy_(self.cached_weight)

        # reengage the autograd engine
        # detach() is necessary to avoid "Trying to backward through the graph a second time" error
        self.cached_weight.detach_()

        # recompute the cache
        # Note: we need the second run to set up the gradients
        new_tensor = self.forward()
        self.cached_weight.copy_(new_tensor)


# class Parametrization(nn.Module, ParametrizationProto):
#     """A parametrization that should be subclassed."""
#
#     parametrized_tensors: dict[str, Tensor]
#     cached_tensors: dict[str, Tensor]
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         # initialize the cache
#         self.cached_tensors = {}
#         self.parametrized_tensors = nn.ParameterDict()
#
#     @abstractmethod
#     def forward(self) -> dict[str, Tensor]:
#         """Update all cached tensors."""
#         ...
#
#     def register_parametrized_tensor(self, name: str, param: nn.Parameter) -> None:
#         """Register a parametrization."""
#         if not isinstance(param, nn.Parameter):
#             raise ValueError("Given tensor is not a nn.Parameter!")
#
#         # register the parametrized tensor.
#         self.parametrized_tensors[name] = param
#
#         # create the cached tensor.
#         self.register_cached_tensor(name, torch.empty_like(param))
#
#         # engage the autograd engine
#         self.cached_tensors[name].copy_(self.parametrized_tensors[name])
#
#     def register_cached_tensor(self, name: str, tensor: Tensor) -> None:
#         """Register a cached tensor."""
#         if name in self.cached_tensors:
#             raise ValueError(f"Cache with {name=!r} already registered!")
#         if name in self.named_buffers():
#             raise ValueError(f"Buffer with {name=!r} already taken!")
#
#         self.register_buffer(name, tensor)
#         self.cached_tensors[name] = getattr(self, name)
#
#     def recompute_cache(self) -> None:
#         # Compute the cached weight matrix
#         new_tensors = self.forward()
#
#         # copy the new tensors into the cache
#         for key, tensor in new_tensors.items():
#             self.cached_tensors[key].copy_(tensor)
#
#     @torch.no_grad()
#     def projection(self) -> None:
#         # update the cached weight matrix
#         self.recompute_cache()
#
#         # copy the cached values into the parametrized tensors
#         for key, tensor in self.parametrized_tensors.items():
#             tensor.copy_(self.cached_tensors[key])
#
#     def reset_cache(self) -> None:
#         # apply projection step.
#         self.projection()
#
#         # reengage the autograd engine
#         # detach() is necessary to avoid "Trying to backward through the graph a second time" error
#         for key, tensor in self.cached_tensors.items():
#             tensor.detach_()
#
#         # recompute the cache
#         # Note: we need the second run to set up the gradients
#         self.recompute_cache()
#
#     def reset_cache_expanded(self) -> None:
#         # apply projection step.
#         with torch.no_grad():
#             # update the cached weight matrix
#             new_tensors = self.forward()
#
#             if new_tensors.keys() != self.cached_tensors.keys():
#                 raise ValueError(
#                     f"{new_tensors.keys()=} != {self.cached_tensors.keys()=}"
#                 )
#
#             # copy the new tensors into the cache
#             for key, tensor in new_tensors.items():
#                 self.cached_tensors[key].copy_(tensor)
#
#         # copy the cached values into the parametrized tensors
#         for key, tensor in self.parametrized_tensors.items():
#             tensor.copy_(self.cached_tensors[key])
#
#         # reengage the autograd engine
#         # detach() is necessary to avoid "Trying to backward through the graph a second time" error
#         for key, tensor in self.cached_tensors.items():
#             tensor.detach_()
#
#         # recompute the cache
#         # Note: we need the second run to set up the gradients
#         # Compute the cached weight matrix
#         new_tensors = self.forward()
#
#         # copy the new tensors into the cache
#         for key, tensor in new_tensors.items():
#             self.cached_tensors[key].copy_(tensor)
#
#
# def parametrize():
#     """Parametrized a single tensor based of a function."""
#     ...
#
#
# def register_parametrization(
#     model: nn.Module,
#     tensor_name: str,
#     parametrization: nn.Module | Callable[[Tensor], Tensor],
#     *,
#     unsafe: bool = False,
# ) -> None:
#     """Drop-in replacement for nn.utils.parametrize.register_parametrization."""
#     ...
#
#
# def get_parametrizations(module: nn.Module, /) -> dict[str, nn.Module]:
#     """Return all parametrizations in a module."""
#     ...
#
#
# def reset_all_caches(module: nn.Module) -> None:
#     """Reset all caches in a module."""
#     for submodule in module.modules():
#         if isinstance(submodule, ParametrizationABC):
#             submodule.reset_cache()
#
#
# class reset_caches(AbstractContextManager):
#     """reset_caches context manager."""
#
#     def __init__(self, module: nn.Module) -> None:
#         self.module = module
#
#     def __enter__(self):
#         reset_all_caches(self.module)
#         return self.module
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         reset_all_caches(self.module)
#         return False
