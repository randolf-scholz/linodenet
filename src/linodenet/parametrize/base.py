"""Parametrizations for Torch.

Methods:
- Parametrization: General purpose parametrization
- SimpleParametrization: Parametrization of a single tensor with a callable
- get_parametrizations: recurisvely returns all parametrizations in a module
- register_parametrization: adds a parametrization to a specific tensor
- cache: context manager which refreshes parametrization cache on exit.
- register_optimizer_hook: automatically adds a hook to optimizer.step() which refreshes the cache after each step.

Usage:
- Create new parametrizations by subclassing Parametrization
- Autogenerate parametrizations from a callable by SimpleParametrization
- add parametrizations to an existing nn.Module by register_parametrization

Issues:
- It would be useful if without caching, the parametrizations would work like simple properties.
    - properties are not supported by JIT...
    - One could disable an if branch
        - but if-branches are slow...
    - context decorator could maybe mutate the nn.Module state...
    - In principle the parametrization only needs to recomputed if the tensor values change,
      so after an optimizer.step() or a reset_parameters() call.

"""

__all__ = [
    # Classes
    "ParametrizationProto",
    "Parametrization",
    "SimpleParametrization",
    # functions
    "register_parametrization",
    "get_parametrizations",
    "reset_all_caches",
]

from abc import abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, jit, nn
from torch.optim import Optimizer


@runtime_checkable
class ParametrizationProto(Protocol):
    """Protocol for parametrizations.

    Note:
        To work with JIT, the listed methods must be annotated with @jit.export.
    """

    @abstractmethod
    def reset_cache(self) -> None:
        """Reset the cached weight matrix."""
        ...

    @abstractmethod
    def _update_cached_tensors(self) -> None:
        """Update cached tensors by recomputing the parametrization."""
        ...

    @abstractmethod
    def projection(self) -> None:
        """Apply the parametrization and copy the result into cache."""
        ...

    @jit.export
    def right_inverse(self) -> None:
        """Compute the right inverse of the parametrization."""
        raise NotImplementedError

    @jit.export
    def reset_parameters(self) -> None:
        """Reapply the initialization."""
        raise NotImplementedError


class Parametrization(nn.Module, ParametrizationProto):
    """A parametrization that should be subclassed.

    Usage:
        # create a model
        model = nn.Linear(4, 4)
        # create a parametrization
        param = Parametrization(model.weight, parametrization)
        # add the parametrization to the model
        model.param = param
        # replace the weight with the parametrized weight
        model.weight = param.parametrized_tensor
    """

    cached_tensors: dict[str, Tensor]
    """DICT: Holds cached tensors."""
    parametrized_tensors: dict[str, Tensor]
    """PARAMDICT: Holds parametrized tensors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the cache
        self.cached_tensors = {}
        self.parametrized_tensors = nn.ParameterDict()

    @abstractmethod
    def forward(self) -> dict[str, Tensor]:
        """Update all tensors based on the current parameters."""
        ...

    def register_parametrized_tensor(
        self, name: str, param: nn.Parameter, /, *, add_to_namespace: bool = True
    ) -> None:
        """Register a parametrization."""
        if not isinstance(param, nn.Parameter):
            raise ValueError("Given tensor is not a nn.Parameter!")
        if name in self.parametrized_tensors:
            raise ValueError(f"Parametrization with {name=!r} already registered!")

        # register the cached tensor.
        self.register_cached_tensor(name, param.clone())
        # self.cached_tensors[name].copy_(param)

        # register the parametrized tensor.
        self.register_parameter(f"original_{name}", param)
        self.parametrized_tensors[name] = param
        assert getattr(self, f"original_{name}") is self.parametrized_tensors[name]

        # engage the autograd engine
        # self.cached_tensors[name].detach_()
        # self.cached_tensors[name].copy_(self.parametrized_tensor[name])

    def register_cached_tensor(self, name: str, tensor: Tensor, /) -> None:
        """Register a cached tensor."""
        if isinstance(tensor, nn.Parameter):
            raise ValueError("Given tensor is a nn.Parameter!")
        if name in self.cached_tensors:
            raise ValueError(f"Cache with {name=!r} already registered!")
        if name in dict(self.named_buffers()):
            raise ValueError(f"Buffer with {name=!r} already taken!")

        self.register_buffer(name, tensor)
        self.cached_tensors[name] = getattr(self, name)

    @jit.export
    def _update_cached_tensors(self) -> None:
        # Compute the cached weight matrix
        new_tensors = self.forward()

        # copy the new tensors into the cache
        for key, tensor in new_tensors.items():
            self.cached_tensors[key].copy_(tensor)

    @jit.export
    def projection(self) -> None:
        """Compute the parametrization and copy the result into parametrized."""
        with torch.no_grad():
            # updated the cached tensors
            self._update_cached_tensors()

            # copy the cached values into the parametrized tensors
            for key, param in self.parametrized_tensors.items():
                param.copy_(self.cached_tensors[key])

    @jit.export
    def reset_cache(self) -> None:
        # apply projection step.
        self.projection()

        # re-engage the autograd engine
        # detach() is necessary to avoid "Trying to backward through the graph a second time" error
        for _, tensor in self.cached_tensors.items():
            tensor.detach_()

        # recompute the cache
        # Note: we need the second run to set up the gradients
        self._update_cached_tensors()

    @jit.export
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
        for _, tensor in self.cached_tensors.items():
            tensor.detach_()

        # recompute the cache
        # Note: we need the second run to set up the gradients
        # Compute the cached weight matrix
        new_tensors = self.forward()

        # copy the new tensors into the cache
        for key, tensor in new_tensors.items():
            self.cached_tensors[key].copy_(tensor)


class SimpleParametrization(nn.Module, ParametrizationProto):
    """Parametrization of a single tensor."""

    # Parameters:
    parametrized_tensor: Tensor
    # Buffers:
    cached_tensor: Tensor

    def __init__(
        self,
        tensor: Tensor,
        parametrization: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()

        # get the tensor to parametrize
        self.register_parameter("parametrized_tensor", tensor)
        self.register_buffer("cached_tensor", torch.empty_like(tensor))

        # get the parametrization
        self._parametrization = parametrization

    def forward(self) -> Tensor:
        """Apply the parametrization to the weight matrix."""
        return self.parametrization(self.parametrized_tensor)

    @jit.export
    def parametrization(self, x: Tensor) -> Tensor:
        """Apply the parametrization."""
        return self._parametrization(x)

    @jit.export
    def _update_cached_tensors(self) -> None:
        # Compute the cached weight matrix
        new_tensor = self.forward()
        self.cached_tensor.copy_(new_tensor)

    @jit.export
    def projection(self) -> None:
        with torch.no_grad():
            # update the cached weight matrix
            self._update_cached_tensors()
            self.parametrized_tensor.copy_(self.cached_tensor)

    @jit.export
    def reset_cache(self) -> None:
        # apply projection step.
        self.projection()

        # re-engage the autograd engine
        # detach() is necessary to avoid "Trying to backward through the graph a second time" error
        self.cached_tensor.detach_()

        # recompute the cache
        # Note: we need the second run to set up the gradients
        self._update_cached_tensors()

    @jit.export
    def reset_cache_expanded(self) -> None:
        with torch.no_grad():
            new_tensor = self.forward()
            self.cached_tensor.copy_(new_tensor)
            self.parametrized_tensor.copy_(self.cached_tensor)

        # reengage the autograd engine
        # detach() is necessary to avoid "Trying to backward through the graph a second time" error
        self.cached_tensor.detach_()

        # recompute the cache
        # Note: we need the second run to set up the gradients
        new_tensor = self.forward()
        self.cached_tensor.copy_(new_tensor)


def register_parametrization(
    model: nn.Module,
    tensor_name: str,
    parametrization: Parametrization | nn.Module | Callable[[Tensor], Tensor],
    *,
    unsafe: bool = False,
) -> None:
    """Drop-in replacement for nn.utils.parametrize.register_parametrization."""
    if hasattr(model, f"{tensor_name}_parametrization"):
        raise NameError(f"{tensor_name}_parametrization already exists!")

    tensor = getattr(model, tensor_name)

    if isinstance(parametrization, nn.Module):
        raise NotImplementedError
    else:
        parametrization = SimpleParametrization(tensor, parametrization)

    # add parametrization to model and rewire the tensor
    setattr(model, f"{tensor_name}_parametrization", parametrization)
    setattr(model, tensor_name, parametrization.parametrized_tensor)


def register_optimizer_hook(optim: Optimizer) -> None:
    """Automatically adds a hook to optimizer.step() which refreshes the cache after each step."""
    raise NotImplementedError


def get_parametrizations(module: nn.Module, /) -> dict[str, nn.Module]:
    """Return all parametrizations in a module."""
    ...


def reset_all_caches(module: nn.Module) -> None:
    """Reset all caches in a module."""
    for submodule in module.modules():
        if isinstance(submodule, ParametrizationProto):
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
