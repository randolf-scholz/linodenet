r"""Alternative to builtin parametrizations of torch.

Goals
-----

- Support for JIT. In particular, we do not use `@property`.
- Class-based parametrizations that allow more complex parametrizations.
    - Example: SpectralNormalization uses an iterative algorithm to compute the spectral norm,
        which is accelerated by caching the singular vectors and reusing them in the next iteration.
- More fine-grained control over what is cached and what is not.
    - In particular, we do not use any global variables

Content
-------

- `Parametrization`: Protocol class for parametrizations.
- `ParametrizationBase`: Parametrization of a single tensor
- `ParametrizationDict`: Parametrization of multiple tensors
- `parametrize`: plug-in replacement for `torch.nn.utils.parametrize`
    wraps a function Tensor -> Tensor into a parametrization.
- `cached`: (quasi) plug-in replacement for `torch.nn.utils.parametrize.cached`
    context manager which refreshes parametrization cache on exit.
- `get_parametrizations`: recursively returns all parametrizations in a module
- `register_parametrization`: adds a parametrization to a specific tensor
- `register_optimizer_hook`: automatically adds a hook to optimizer.step() which refreshes the cache after each step.

Differences
-----------

- Instead of inserting properties, we use buffers, because JIT does not support properties.
  This means that the parametrization is not recomputed automatically when the original tensor changes.
  Instead, the parametrization needs to be recomputed manually by calling `update_parametrization()`.
- register_parametrization is intended as a drop-in replacement for
  `torch.nn.utils.parametrize.register_parametrization`.
  However, it is not equivalent. In particular, it does not support replacing a tensor with
  other tensors. For example, a rank-one parametrization is realized by projecting onto the
  low rank manifold in the forward pass and projecting back to the full rank manifold when
  updating the parameters. This is important to ensure parametrizations are chainable and to maintain
  type-safety.

Usage
-----

- Create new parametrizations by subclassing Parametrization
- Autogenerate parametrizations from a callable by SimpleParametrization
- add parametrizations to an existing nn.Module by register_parametrization

Issues
------

- It would be useful if without caching, the parametrizations would work like simple properties.
- properties are not supported by JIT...
- One could disable an if branch
    - but if-branches are slow...
- context decorator could maybe mutate the nn.Module state...
- In principle the parametrization only needs to recomputed if the tensor values change,
  so after an optimizer.step() or a reset_parameters() call.
- Currently unsupported to use multiple parametrizations on the same tensor.

Classes
-------

- `ParametrizationProto`: Protocol for all parametrizations.
- `Parametrization`: Base class for parametrizations that maintain a single cached tensor.
    - `parametrize`: wraps a function Tensor -> Tensor into a parametrization.
    - `ParametrizationCache`: Base class for parametrization with multiple cached tensors.
- `ParametrizationDict`: Base class for complex parametrization with multiple parametrized and cached tensors.
"""

__all__ = [
    # Protocol
    "Surjection",
    "Parametrization",
    "BoundParametrization",
    # Classes
    "WithoutRightInverse",
    "ParametrizationBase",
    "WrappedParametrization",
    "ParametrizationList",
    # torch.nn.utils.parametrize replacements
    "parametrize",
    "is_parametrized",
    "register_parametrization",
    "remove_parametrizations",
    "cached",
    # Functions
    "assert_is_safe_parametrization",
    "deepcopy_with_parametrizations",
    # "detach_caches",
    "get_parametrizations",
    "iter_parametrizations",
    "is_parametrization",
    "register_optimizer_hook",
    # "update_caches",
    # "update_originals",
    "update_parametrizations",
]

import copy
import warnings
from abc import abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager, ContextDecorator
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Never,
    Optional,
    Protocol,
    Self,
    TypeIs,
    cast,
    get_protocol_members,
    runtime_checkable,
)
from warnings import deprecated

import torch
from torch import Tensor, jit, nn
from torch.optim import Optimizer

from linodenet.containers import ModuleSequence


@runtime_checkable
class Surjection[X, Y](Protocol):
    r"""A protocol for surjections.

    Surjections are maps that are onto, i.e., for every $y$ in $Y$, there exists
    an $x$ in $X$ such that $f(x) = y$. In particular, they admit a right inverse,
    i.e., a map $g: Y -> X$ such that $f(g(y)) = y$ for all $y$ in $Y$.
    """

    @abstractmethod
    def forward(self, x: X, /) -> Y: ...
    @abstractmethod
    def right_inverse(self, y: Y, /) -> X: ...


@runtime_checkable
class Parametrization(Protocol):
    r"""Protocol for parametrizations."""

    @abstractmethod
    def forward(self, arg: Tensor, /) -> Tensor:
        r"""Apply the parametrization to the given tensor."""
        ...

    @abstractmethod
    def right_inverse(self, arg: Tensor, /) -> Tensor | None:
        r"""Compute the right inverse of the parametrization."""
        ...

    @abstractmethod
    def update_parametrization(self) -> None:
        r"""Update both the cached and the original tensors."""
        ...


class WithoutRightInverse(nn.Module):
    r"""Wrapper for parametrizations without right inverse."""

    def __init__(
        self,
        parametrization: nn.Module,
    ) -> None:
        super().__init__()
        self.parametrization = parametrization

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        return self.parametrization(x)

    @jit.export
    def right_inverse(self, _: Tensor) -> Tensor | None:
        return None


class ParametrizationList(ModuleSequence):
    r"""TODO: implement ParametrizationList."""


@deprecated("do not use")
class BoundParametrization(Protocol):
    r"""Protocol for parametrizations that wrap a single tensor.

    Note:
        To work with JIT, the listed methods must be annotated with @jit.export.
        - "wrapped tensor" refers to the tensor that is wrapped by the parametrization.
        - "cached tensor" refers to the tensor that is used to cache the parametrization.
        - We do not add attributes to the parametrization to avoid making a Data Protocol.
    """

    @abstractmethod
    def __init__(self, tensor: Tensor, /) -> None:
        r"""Initialize the parametrization.

        Args:
            tensor: The tensor to parametrize.
        """
        ...

    @abstractmethod
    def forward(self, arg: Tensor, /) -> Tensor:
        r"""Apply the parametrization to the given tensor."""
        ...

    cached_parameter: Tensor
    r"""BUFFER: Holds cached version of the parametrized tensor."""
    original_parameter: nn.Parameter
    r"""PARAM: Holds parametrized tensors."""

    @abstractmethod
    def apply_parametrization(self) -> Any: ...
    @abstractmethod
    def get_original_tensor(self) -> Tensor: ...
    @abstractmethod
    def get_cached_tensor(self) -> Tensor: ...

    # mixin  methods
    @jit.export
    def right_inverse(self, y: Tensor, /) -> Tensor | None:
        r"""Compute the right inverse of the parametrization.

        Returns:
            Tensor | None: The pullback of the original tensor, or None if not implemented.

        The right inverse is such that `parametrization(right_inverse(y)) == y`.
        I.e. starting from an already parametrized tensor, the right inverse
        will return the original tensor. This is needed when the original tensor
        already has a parametrization applied to it and hence belongs to some
        constraint set.

        Here, we default to the identity function, which is correct for projections,
        since projections are idempotent. ($y = f(x) ⟹ f(id(y)) = f(y) = f(f(x)) = f(x) = y$)
        """
        return y

    @jit.export
    def detach_cache(self) -> None:
        r"""Detach the cached tensors from the autograd engine.

        This method should be called after `update_original()` to avoid
        "Trying to backward through the graph a second time" error.
        """
        self.get_cached_tensor().detach_()

    @jit.export
    def update_cache(self) -> None:
        r"""Update the cached tensors by recomputing the parametrization using the original tensors.

        Note:
            This method should use inplace `copy_` operations to update the cached tensors.
        """
        new_tensor = self.apply_parametrization()
        self.get_cached_tensor().copy_(new_tensor)

    @jit.export
    @torch.no_grad()
    def update_original(self) -> None:
        r"""Update the original tensors based on the cached tensors.

        Note:
            This method should use inplace `copy_` operations to update the original tensors.
            This method should always be called with `torch.no_grad()`.
        """
        # Call `right_inverse` to get the pullback of the original tensor.
        pullback = self.right_inverse(self.get_cached_tensor())
        if pullback is not None:
            # Use inplace `copy_` operations to update the original tensors.
            self.get_original_tensor().copy_(pullback)

    @jit.export
    @torch.no_grad()
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
        # ①. recompute the parametrization given the modified parameters
        self.update_cache()

        # ②. update the original parameters based on the new parametrization
        self.update_original()

        # ③. detach the cached tensors from the autograd engine
        # This method should be called after `update_original()` to avoid
        #         "Trying to backward through the graph a second time" error.
        self.detach_cache()

        # ④. re-enable the autograd engine
        with torch.enable_grad():
            self.update_cache()


def is_parametrization(obj: Any, /) -> TypeIs[Parametrization]:
    r"""Check if the object is a Parametrization.

    This method is needed because standard isinstance checks do not work with jit.ScriptModule.
    """
    return all(hasattr(obj, member) for member in get_protocol_members(Parametrization))


# region base classes ------------------------------------------------------------------
class _WithPostInitMeta(type):
    @staticmethod
    def __post_init__(_: Never, /) -> None:
        pass

    def __call__[T](cls: type[T], *args: Any, **kwargs: Any) -> T:
        instance = super().__call__(*args, **kwargs)  # type: ignore[misc]
        instance.__post_init__()
        return instance

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwargs: Any,
    ) -> type:
        new: type[Any] = super().__new__(cls, name, bases, namespace, **kwargs)
        if getattr(new, "__post_init__", None) is None:
            namespace["__post_init__"] = _WithPostInitMeta.__post_init__
            new = super().__new__(cls, name, bases, namespace, **kwargs)
        return new


class ParametrizationBase(nn.Module, metaclass=_WithPostInitMeta):
    r"""Base class for parametrization of a single tensor using a single cached tensor."""

    original_parameter: nn.Parameter
    r"""PARAM: Holds parametrized tensors."""
    cached_parameter: Tensor
    r"""BUFFER: Holds cached version of the parametrized tensor."""
    unsafe: Final[bool]

    def __init__(
        self,
        tensor: Tensor,
        *,
        unsafe: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(tensor, nn.Parameter):
            raise TypeError("tensor must be a nn.Parameter")

        # get the tensor to parametrize
        self.register_parameter("original_parameter", tensor)
        self.register_buffer("cached_parameter", tensor.clone().detach())
        self.unsafe = unsafe

    # TODO: use this instead of metaclass?
    r"""
    def __init_subclass__(cls: type[Self], **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        import functools
        original_init = cls.__init__

        # add __post_init__ hook to __init__
        @functools.wraps(original_init)
        def __init_with_post_init(self: Self, /, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            self.__post_init__()

        cls.__init__ = __init_with_post_init  # type: ignore[assignment]  # pyright: ignore[reportAttributeAccessIssue]
    """

    def __post_init__(self) -> None:
        if not self.unsafe:
            assert_is_safe_parametrization(self, self.original_parameter)

        self.update_cache()

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor: ...
    @abstractmethod
    def right_inverse(self, y: Tensor, /) -> Tensor | None: ...

    # implement protocol methods:
    @jit.export
    def apply_parametrization(self) -> Tensor:
        r"""Apply the parametrization to the weight matrix."""
        return self.forward(self.original_parameter)

    @jit.export
    def update_cache(self) -> None:
        r"""Update the cached tensors by recomputing the parametrization using the original tensors.

        Note:
            This method should use inplace `copy_` operations to update the cached tensors.
        """
        new_tensor = self.apply_parametrization()
        self.cached_parameter.copy_(new_tensor)

    @jit.export
    def get_original_tensor(self) -> Tensor:
        r"""Get the original tensor from the cached tensor."""
        return self.original_parameter

    @jit.export
    def get_cached_tensor(self) -> Tensor:
        r"""Get the cached tensor from the original tensor."""
        return self.cached_parameter

    @jit.export
    def detach_cache(self) -> None:
        self.cached_parameter.detach_()

    @jit.export
    @torch.no_grad()
    def update_original(self) -> None:
        pullback = self.right_inverse(self.cached_parameter)
        if pullback is not None:
            self.original_parameter.copy_(pullback)

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


class WrappedParametrization(ParametrizationBase):
    r"""Base class for parametrization of a single tensor using a single cached tensor."""

    original_parameter: nn.Parameter
    r"""PARAM: Holds parametrized tensors."""
    cached_parameter: Tensor
    r"""BUFFER: Holds cached version of the parametrized tensor."""

    def __init__(
        self,
        tensor: Tensor,
        parametrization: nn.Module,
        *,
        unsafe: bool = False,
    ) -> None:
        super().__init__(tensor, unsafe=unsafe)

        if not callable(getattr(parametrization, "right_inverse", None)):
            parametrization = WithoutRightInverse(parametrization)

        if TYPE_CHECKING:
            assert (  # noqa: PT018
                isinstance(parametrization, nn.Module)
                and isinstance(parametrization, Surjection)
            )
        self.parametrization: Surjection = parametrization.to(
            device=tensor.device, dtype=tensor.dtype
        )

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the parametrization."""
        return self.parametrization.forward(x)

    @jit.export
    def right_inverse(self, y: Tensor) -> Tensor | None:
        return self.parametrization.right_inverse(y)


def parametrize(
    tensor: Tensor,
    parametrization: nn.Module | type[ParametrizationBase],
    *,
    unsafe: bool = False,
) -> ParametrizationBase:
    if isinstance(parametrization, nn.Module):
        return WrappedParametrization(tensor, parametrization, unsafe=unsafe)
    if isinstance(parametrization, type):
        return parametrization(tensor)
    raise TypeError(
        "parametrization must be either a nn.Module or a Parametrization class"
    )


# endregion base classes ---------------------------------------------------------------


# region torch parametrize replacements  -----------------------------------------------
def register_parametrization(
    module: nn.Module,
    tensor_name: str,
    parametrization: nn.Module | type[ParametrizationBase],
    *,
    unsafe: bool = False,
) -> None:
    r"""Drop-in replacement for nn.utils.parametrize.register_parametrization."""
    tensor = getattr(module, tensor_name)
    if not isinstance(tensor, nn.Parameter):
        raise TypeError(f"{tensor_name} is not a parameter!")

    if tensor_name in getattr(module, "parametrizations", {}):
        raise NameError(f"{tensor_name} already parametrized!")

    wrapper = parametrize(tensor, parametrization, unsafe=unsafe)

    if not isinstance(wrapper, nn.Module) or not is_parametrization(wrapper):
        raise TypeError(f"{parametrization} does not produce a valid parametrization!")

    if not unsafe:
        assert_is_safe_parametrization(wrapper, tensor)

    # add the parametrization to model.parametrizations ModuleDict
    match ps := getattr(module, "parametrizations", None):
        case None:
            module.register_module(
                "parametrizations",
                nn.ModuleDict({tensor_name: wrapper}),
            )
        case nn.ModuleDict() as parametrizations:
            parametrizations[tensor_name] = wrapper
        case _:
            raise TypeError(f"Expected a nn.ModuleDict, but got {type(ps)}!")

    # add the original tensor to model.parametrized_tensors ParameterDict
    match ts := getattr(module, "parametrized_tensors", None):
        case None:
            module.register_module(
                "parametrized_tensors",
                nn.ParameterDict({tensor_name: wrapper.original_parameter}),
            )
        case nn.ParameterDict() as parametrized_tensors:
            parametrized_tensors[tensor_name] = wrapper.original_parameter
        case _:
            msg = f"Expected a nn.ParameterDict, but got {type(ts)}!"
            raise TypeError(msg)

    # add a buffer in place of the original tensor
    delattr(module, tensor_name)
    module.register_buffer(tensor_name, wrapper.get_cached_tensor())

    # initialize the parametrization
    wrapper.update_parametrization()


def is_parametrized(module: nn.Module, tensor_name: Optional[str] = None) -> bool:
    r"""Return True if the module has any parametrizations."""
    if tensor_name is None:
        return any(is_parametrization(m) for m in module.modules())

    parametrizations = get_parametrizations(module)
    return tensor_name in parametrizations


def assert_is_safe_parametrization(
    parametrization: BoundParametrization, tensor: Tensor
) -> None:
    r"""Check if the parametrization is safe to apply to the tensor."""
    Y = tensor
    X = parametrization.forward(Y)
    if not isinstance(X, Tensor):
        raise TypeError(
            f"A parametrization must return a tensor. Got {type(X).__name__}."
        )
    if X.dtype != Y.dtype:
        raise ValueError(
            "A parametrization may not change the dtype of the tensor,"
            " unless the `unsafe` flag is enabled."
            f"\noriginal dtype:     {Y.dtype}"
            f"\nparametrized dtype: {X.dtype}"
        )
    if X.shape != Y.shape:
        raise ValueError(
            "A parametrization may not change the shape of the tensor,"
            " unless the `unsafe` flag is enabled."
            f"\n original shape:     {Y.shape}"
            f"\n parametrized shape: {X.shape}"
        )
    # check right inverse
    try:
        Z = parametrization.right_inverse(X)
    except NotImplementedError:
        pass
    else:
        if Z is None:
            return
        if not isinstance(Z, Tensor):
            raise TypeError(
                f"right_inverse must return a tensor. Got: {type(Z).__name__}"
            )
        if Z.dtype != Y.dtype:
            raise ValueError(
                "The tensor returned by right_inverse must have the same dtype,"
                " unless the `unsafe` flag is enabled."
                f"\noriginal dtype: {Y.dtype}"
                f"\nreturned dtype: {Z.dtype}"
            )
        if Z.shape != Y.shape:
            raise ValueError(
                "The tensor returned by right_inverse must have the same shape,"
                " unless the `unsafe` flag is enabled."
                f"\noriginal shape: {Y.shape}"
                f"\nreturned shape: {Z.shape}"
            )


def get_parametrizations(module: nn.Module, /) -> nn.ModuleDict:
    r"""Return all parametrizations in a module."""
    match ps := getattr(module, "parametrizations", None):
        case None:
            return nn.ModuleDict()
        case nn.ModuleDict() as parametrizations:
            return parametrizations
        case jit.RecursiveScriptModule() as parametrizations:  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateImportUsage]
            warnings.warn(
                "Scripted module! Not all functionality may be available.", stacklevel=2
            )
            return cast("nn.ModuleDict", parametrizations)
        case _:
            raise TypeError(f"Expected a nn.ModuleDict, but got {type(ps)}!")


def remove_parametrizations(
    module: nn.Module,
    tensor_name: str,
    *,
    leave_parametrized: bool = True,
) -> nn.Module:
    r"""Remove the parametrizations on a tensor in a module."""
    raise NotImplementedError


class cached(ContextDecorator, AbstractContextManager):
    r"""Context Manager to update the caches of all the given modules."""

    def __init__(self, *modules: *tuple[nn.Module, *tuple[nn.Module, ...]]) -> None:
        self.modules = modules
        if not self.modules:
            raise ValueError("At least one module must be provided!")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
        /,
    ) -> Literal[False]:
        for module in self.modules:
            update_parametrizations(module)

        return False


# endregion torch parametrize replacements ---------------------------------------------


# region functions for parametrization -------------------------------------------------
def iter_parametrizations(module: nn.Module, /) -> Iterator[Parametrization]:
    r"""Yields all parametrizations in a module."""
    for m in module.modules():
        if is_parametrization(m):
            yield m


def update_parametrizations(module: nn.Module, /) -> None:
    r"""Update all parametrizations in a module."""
    for parametrization in iter_parametrizations(module):
        parametrization.update_parametrization()


#
# def detach_caches(module: nn.Module, /) -> None:
#     r"""Detach all caches in a module."""
#     for parametrization in iter_parametrizations(module):
#         parametrization.detach_cache()
#
#
# def update_originals(module: nn.Module, /) -> None:
#     r"""Update all original tensors in a module."""
#     for parametrization in iter_parametrizations(module):
#         parametrization.update_original()
#
#
# def update_caches(module: nn.Module, /) -> None:
#     r"""Update all cached tensors in a module."""
#     for parametrization in iter_parametrizations(module):
#         parametrization.update_cache()

# endregion functions for parametrization ----------------------------------------------


# region additional functions ----------------------------------------------------------
def register_optimizer_hook(
    optim: Optimizer, /, *module_or_param: nn.Module | Parametrization
) -> None:
    r"""Automatically adds a hook to `optimizer.step()` which refreshes the cache after each step."""
    # collect all parametrizations
    parametrizations = []
    for module in module_or_param:
        if is_parametrization(module):
            parametrizations.append(module)
        else:
            parametrizations.extend(iter_parametrizations(module))

    def hook(opt: Optimizer, /, *args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        r"""Hook to update the parametrization after each optimizer step."""
        for parametrization in parametrizations:
            parametrization.update_parametrization()

    optim.register_step_post_hook(hook)


def deepcopy_with_parametrizations[M: nn.Module](module: M, /) -> M:
    r"""Deepcopy a module."""
    # detach all caches
    # detach_caches(module)
    with torch.no_grad():
        # deepcopy the module
        cloned = copy.deepcopy(module)

    # recompute all caches for the original module
    # update_parametrizations(module)

    # recompute all caches for the cloned module
    update_parametrizations(cloned)
    # return the cloned module
    return cloned


# endregion additional functions -------------------------------------------------------
