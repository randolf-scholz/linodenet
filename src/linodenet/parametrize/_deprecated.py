r"""Deprecated parametrization protocols and base classes."""

__all__ = [
    "GeneralParametrization",
    "ParametrizationDict",
    "ParametrizationMulticache",
]

from abc import abstractmethod
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, jit, nn

from linodenet.parametrize.base import Parametrization


@runtime_checkable
class GeneralParametrization(Protocol):
    r"""Protocol for parametrizations.

    In most cases, use `Parametrization` instead of this protocol.
    This protocol is only useful if you want to parametrize multiple tensors simultaneously.

    Note:
        To work with JIT, the listed methods must be annotated with @jit.export.
        - "wrapped tensor" refers to the tensor that is wrapped by the parametrization.
        - "cached tensor" refers to the tensor that is used to cache the parametrization.

    Warnings:
        # SEE: https://github.com/pytorch/pytorch/pull/103001
        Parametrization can cause `deepcopy` to fail. To use deepcopy:
        1. Call `detach_cache()` to detach the cached tensors from the autograd engine.
        2. Call `deepcopy` on the model.
        3. Call `update_cache()` to re-enable the autograd engine.
    """

    @abstractmethod
    def apply_parametrization(self) -> Any:
        r"""Compute the parametrization, takes NO parameters."""
        ...

    @abstractmethod
    def update_cache(self) -> None:
        r"""Update the cached tensors by recomputing the parametrization using the original tensors.

        Note:
            This method should use inplace `copy_` operations to update the cached tensors.
        """
        ...

    @abstractmethod
    def update_original(self) -> None:
        r"""Update the original tensors based on the cached tensors.

        Note:
            This method should use inplace `copy_` operations to update the original tensors.
            This method should always be called with `torch.no_grad()`.
        """
        ...

    @abstractmethod
    def detach_cache(self) -> None:
        r"""Detach the cached tensors from the autograd engine.

        This method should be called after `update_original()` to avoid
        "Trying to backward through the graph a second time" error.
        """
        ...

    @jit.export
    def update_parametrization(self) -> None:
        r"""Update both the cached and the original tensors.

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


class ParametrizationMulticache(Parametrization):
    r"""Base class for parametrizations that maintain additional cached tensors."""

    original_parameter: nn.Parameter
    r"""PARAM: Holds parametrized tensors."""
    cached_parameter: Tensor
    r"""BUFFER: Holds cached version of the parametrized tensor."""
    cached_tensors: dict[str, Tensor]  # NOTE: cannot use nn.ParameterDict due to JIT
    r"""BUFFER-DICT: Holds auxiliary cached tensors."""

    def __init__(self, tensor: Tensor, /) -> None:
        super().__init__(tensor)
        # get the tensor to parametrize
        # if not isinstance(tensor, nn.Parameter):
        #     raise TypeError("tensor must be a nn.Parameter")

        # self.register_parameter("original_parameter", tensor)
        # self.register_buffer("cached_parameter", tensor.clone().detach())

        # Q: Use nn.BufferDict? https://github.com/pytorch/pytorch/issues/37386
        self.cached_tensors = {}

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the parametrization.

        Should return a tuple of the parametrized tensor and a dictionary of auxiliary tensors.
        """
        ...

    @jit.export
    def apply_parametrization(self) -> Tensor:
        r"""Apply the parametrization to the weight matrix."""
        return self.forward(self.original_parameter)

    @jit.export
    def detach_cache(self) -> None:
        self.cached_parameter.detach_()
        # detach all auxiliary cached tensors
        for tensor in self.cached_tensors.values():
            tensor.detach_()

    def register_cached_tensor(self, name: str, tensor: Tensor, /) -> None:
        r"""Register a cached tensor."""
        if isinstance(tensor, nn.Parameter):
            raise TypeError("Given tensor is a nn.Parameter!")
        if name in self.cached_tensors:
            raise ValueError(f"Cache with {name=!r} already registered!")
        if name in dict(self.named_buffers()):
            raise ValueError(f"Buffer with {name=!r} already taken!")

        self.register_buffer(name, tensor)
        self.cached_tensors[name] = getattr(self, name)


# FIXME: use MutableMapping https://github.com/pytorch/pytorch/issues/110959
class ParametrizationDict(nn.Module, GeneralParametrization):
    r"""Base class for parametrizations that maintain a dictionary of parametrized tensors.

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
    r"""DICT: Holds all cached tensors."""
    parametrized_tensors: dict[str, nn.Parameter]
    r"""DICT: Holds parametrized tensors."""

    def __init__(self) -> None:
        super().__init__()

        # initialize the cache
        self.cached_tensors = {}  # TODO: Use nn.BufferDict?
        self.parametrized_tensors = {}  # NOTE: JIT error with nn.ParameterDict.

    def __iter__(self) -> Iterator[str]:
        return iter(self.parametrized_tensors)

    def __len__(self) -> int:
        return len(self.parametrized_tensors)

    def __getitem__(self, item: str, /) -> nn.Parameter:
        return self.parametrized_tensors[item]

    def __setitem__(self, key: str, value: nn.Parameter, /) -> None:
        self.register_parametrized_tensor(key, value)

    def __delitem__(self, key: str, /) -> None:
        del self.parametrized_tensors[key]
        del self.cached_tensors[key]
        delattr(self, key)

    @abstractmethod
    def apply_parametrization(self) -> dict[str, Tensor]:
        r"""Update all tensors based on the current parameters."""
        ...

    @jit.export
    def update_cache(self) -> None:
        new_tensors = self.apply_parametrization()
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
        r"""Register a cached tensor."""
        if isinstance(tensor, nn.Parameter):
            raise TypeError("Given tensor is a nn.Parameter!")
        if name in self.cached_tensors:
            raise ValueError(f"Cache with {name=!r} already registered!")
        if name in dict(self.named_buffers()):
            raise ValueError(f"Buffer with {name=!r} already taken!")

        self.register_buffer(name, tensor)
        self.cached_tensors[name] = getattr(self, name)

    def register_parametrized_tensor(self, name: str, param: nn.Parameter, /) -> None:
        r"""Register a parametrization."""
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

        if getattr(self, f"original_{name}") is not self.parametrized_tensors[name]:
            raise ValueError(f"original_{name} is not the same as {name}!")

        # engage the autograd engine
        # self.cached_tensors[name].detach_()
        # self.cached_tensors[name].copy_(self.parametrized_tensor[name])
