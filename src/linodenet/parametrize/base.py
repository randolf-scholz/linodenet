"""Alternative to builtin parametrizations of torch.

Goals:
    - Support for JIT. In particular, we do not use `@property`.
    - Class-based parametrizations that allow more complex parametrizations.
        - Example: SpectralNormalization uses an iterative algorithm to compute the spectral norm,
            which is accelerated by caching the singular vectors and reusing them in the next iteration.
    - More fine grained control over what is cached and what is not.
        - In particular, we do not use any global variables

Content:
    - `Parametrization`: Protocol class for parametrizations.
    - `ParametrizationBase`: Parametrization of a single tensor
    - `ParametrizationDict`: Parametrization of multiple tensors
    - `parametrize`: plug-in replacement for `torch.nn.utils.parametrize`
        wraps a function Tensor -> Tensor into a parametrization.
    - `cached`: (quasi) plug-in replacement for `torch.nn.utils.parametrize.cached`
        context manager which refreshes parametrization cache on exit.

    - get_parametrizations: recursively returns all parametrizations in a module
    - register_parametrization: adds a parametrization to a specific tensor
    - register_optimizer_hook: automatically adds a hook to optimizer.step() which refreshes the cache after each step.

Differences:
    - By default, parametrizations


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

Classes:
    - `ParametrizationProto`: Protocol for all parametrizations.
    - `Parametrization`: Base class for parametrizations that maintain a single cached tensor.
        - `parametrize`: wraps a function Tensor -> Tensor into a parametrization.
        - `ParametrizationCache`: Base class for parametrization with multiple cached tensors.
    - `ParametrizationDict`: Base class for complex parametrization with multiple parametrized and cached tensors.
"""

__all__ = [
    # Protocol
    "Parametrization",
    # Classes
    "ParametrizationDict",
    "ParametrizationBase",
    "parametrize",
    # Functions
    "register_parametrization",
    "get_parametrizations",
    "reset_all_caches",
]

from abc import abstractmethod
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, jit, nn
from torch.optim import Optimizer


@runtime_checkable
class Parametrization(Protocol):
    """Protocol for parametrizations.

    Note:
        To work with JIT, the listed methods must be annotated with @jit.export.
        - "wrapped tensor" refers to the tensor that is wrapped by the parametrization.
        - "cached tensor" refers to the tensor that is used to cache the parametrization.
    """

    @abstractmethod
    def parametrization(self) -> Any:
        """Compute the parametrization, takes NO parameters."""
        ...

    @abstractmethod
    def update_cache(self) -> None:
        """Update the cached tensors by recomputing the parametrization using the original tensors.

        Note:
            This method should use inplace `copy_` operations to update the cached tensors.
        """
        ...

    @abstractmethod
    def update_original(self) -> None:
        """Update the original tensors based on the cached tensors.

        Note:
            This method should use inplace `copy_` operations to update the original tensors.
            This method should always be called with `torch.no_grad()`.
        """
        ...

    @abstractmethod
    def detach_cache(self) -> None:
        """Detach the cached tensors from the autograd engine.

        This method should be called after `update_original()` to avoid
        "Trying to backward through the graph a second time" error.
        """
        ...

    @jit.export
    def update_parametrization(self) -> None:
        """Update both the cached and the original tensors.

        This function needs to be called after each `optimizer.step()` call.
        Internally, it should perform the following steps:

        1. Call `update_cache()` **without gradients**
            to get the new parametrization given the modified parameters.
        2. Call `update_original()` **without gradients**
            to update the original parameters based on the new parametrization.
        3. Call `detach_cache()` to detach the cached tensors from the autograd engine.
        4. Call `update_cache()` a second time **with gradients** to re-enable the autograd engine.
        """
        with torch.no_grad():
            # recompute the parametrization given the modified parameters
            self.update_cache()

            # update the original parameters based on the new parametrization
            self.update_original()

            # detach the cached tensors from the autograd engine
            self.detach_cache()

        # re-enable the autograd engine
        self.update_cache()


class ParametrizationBase(nn.Module, Parametrization):
    """Base class for parametrization of a single tensor using a single cached tensor."""

    original_parameter: Tensor
    """PARAM: Holds parametrized tensors."""
    cached_parameter: Tensor
    """BUFFER: Holds cached version of the parametrized tensor."""

    def __init__(self, tensor: Tensor) -> None:
        super().__init__()

        # get the tensor to parametrize
        assert isinstance(tensor, nn.Parameter), "tensor must be a parameter"
        self.register_parameter("original_parameter", tensor)
        self.register_buffer("cached_parameter", tensor.clone().detach())

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply the parametrization."""
        ...

    @jit.export
    def parametrization(self) -> Tensor:
        """Apply the parametrization to the weight matrix."""
        return self.forward(self.original_parameter)

    @jit.export
    def update_cache(self) -> None:
        new_tensor = self.parametrization()
        self.cached_parameter.copy_(new_tensor)

    @jit.export
    @torch.no_grad()
    def update_original(self) -> None:
        self.original_parameter.copy_(self.cached_parameter)

    @jit.export
    @torch.no_grad()
    def detach_cache(self) -> None:
        self.cached_parameter.detach_()

    # @jit.export
    # def update_parametrization(self) -> None:
    #     # apply projection step.
    #     self.update_original()
    #
    #     # re-engage the autograd engine
    #     # detach() is necessary to avoid "Trying to backward through the graph a second time" error
    #     self.cached_parameter.detach_()
    #
    #     # recompute the cache
    #     # Note: we need the second run to set up the gradients
    #     self.update_cache()


class parametrize(ParametrizationBase):
    """Parametrization of a single tensor."""

    original_parameter: Tensor
    """PARAM: Holds parametrized tensors."""
    cached_parameter: Tensor
    """BUFFER: Holds cached version of the parametrized tensor."""

    def __init__(
        self, tensor: Tensor, parametrization: Callable[[Tensor], Tensor] | nn.Module
    ) -> None:
        super().__init__(tensor)
        self._parametrization = parametrization

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        """Apply the parametrization."""
        return self._parametrization(x)


# FIXME: use MutableMapping https://github.com/pytorch/pytorch/issues/110959
class ParametrizationDict(nn.Module, Parametrization):
    """Base class for parametrizations that maintain a dictionary of parametrized tensors.

    Example:
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
    """DICT: Holds all cached tensors."""
    parametrized_tensors: dict[str, nn.Parameter]
    """DICT: Holds parametrized tensors."""

    def __init__(self) -> None:
        super().__init__()

        # initialize the cache
        self.cached_tensors = {}  # Q: Use nn.BufferDict?
        self.parametrized_tensors = {}  # NOTE: JIT error with nn.ParameterDict.

    def __iter__(self) -> Iterator[str]:
        return iter(self.parametrized_tensors)

    def __len__(self) -> int:
        return len(self.parametrized_tensors)

    def __getitem__(self, item: str) -> nn.Parameter:
        return self.parametrized_tensors[item]

    def __setitem__(self, key: str, value: nn.Parameter) -> None:
        self.register_parametrized_tensor(key, value)

    def __delitem__(self, key: str) -> None:
        del self.parametrized_tensors[key]
        del self.cached_tensors[key]
        delattr(self, key)

    @abstractmethod
    def parametrization(self) -> dict[str, Tensor]:
        """Update all tensors based on the current parameters."""
        ...

    @jit.export
    def update_cache(self) -> None:
        new_tensors = self.parametrization()
        for key, tensor in new_tensors.items():
            self.cached_tensors[key].copy_(tensor)

    @jit.export
    @torch.no_grad()
    def update_original(self) -> None:
        for key, param in self.parametrized_tensors.items():
            param.copy_(self.cached_tensors[key])

    @jit.export
    @torch.no_grad()
    def detach_cache(self) -> None:
        for tensor in self.cached_tensors.values():
            tensor.detach_()

    def register_cached_tensor(self, name: str, tensor: Tensor, /) -> None:
        """Register a cached tensor."""
        if isinstance(tensor, nn.Parameter):
            raise TypeError("Given tensor is a nn.Parameter!")
        if name in self.cached_tensors:
            raise ValueError(f"Cache with {name=!r} already registered!")
        if name in dict(self.named_buffers()):
            raise ValueError(f"Buffer with {name=!r} already taken!")

        self.register_buffer(name, tensor)
        self.cached_tensors[name] = getattr(self, name)

    def register_parametrized_tensor(
        self, name: str, param: nn.Parameter, /, *, add_to_namespace: bool = True
    ) -> None:
        """Register a parametrization."""
        if not isinstance(param, nn.Parameter):
            raise TypeError("Given tensor is not a nn.Parameter!")
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


def register_parametrization(
    model: nn.Module,
    tensor_name: str,
    parametrization: ParametrizationDict | nn.Module | Callable[[Tensor], Tensor],
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
        parametrization = parametrize(tensor, parametrization)

    # add parametrization to model and rewire the tensor
    setattr(model, f"{tensor_name}_parametrization", parametrization)
    setattr(model, tensor_name, parametrization.parametrized_tensor)


def register_optimizer_hook(optim: Optimizer, /) -> None:
    """Automatically adds a hook to optimizer.step() which refreshes the cache after each step."""
    raise NotImplementedError(optim)


def get_parametrizations(module: nn.Module, /) -> Iterator[Parametrization]:
    """Return all parametrizations in a module."""
    for m in module.modules():
        if isinstance(m, Parametrization):
            yield m


def reset_all_caches(module: nn.Module) -> None:
    """Reset all caches in a module."""
    for submodule in module.modules():
        if isinstance(submodule, Parametrization):
            submodule.update_parametrization()


class cached(AbstractContextManager):
    """Context Manager to update the caches of all the given modules."""

    def __init__(self, *modules: nn.Module) -> None:
        self.modules = modules

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module in self.modules:
            for param in get_parametrizations(module):
                param.update_parametrization()

        return False
