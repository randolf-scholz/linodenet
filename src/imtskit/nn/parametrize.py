r"""Cached parametrizations for PyTorch module parameters.

## Cache behavior

Parametrized attributes are buffers backed by `ParametrizationList` caches.
The trainable source remains in `original_parameter`. Attribute access reads
the cache rather than recomputing the parametrization.

Call `update_parametrizations` after modifying parameters, or register an
optimizer post-step hook with `register_optimizer_hook`. The `cached` context
manager refreshes the supplied modules and their descendants on exit.
Accumulating gradients in an original parameter marks its cache stale and
fills it with NaNs until it is refreshed.

## API

- `Surjection`: Protocol for maps with an optional right inverse.
- `Parametrization`: Protocol for maps with a right inverse and cache update.
- `Parametrized`: Protocol for modules exposing a `parametrizations` mapping.
- `ParametrizationList`: Chain of parametrizations sharing one source parameter
    and one cached output tensor.
- `register_parametrization`: Register a map or append it to an existing chain.
- `get_parametrizations`: Read a module's own parametrization mapping.
- `update_parametrizations`: Refresh parametrizations recursively.
- `register_optimizer_hook`: Refresh caches after each optimizer step.
- `cached`: Context manager and decorator that refreshes caches on exit.
- `remove_parametrizations`: Placeholder; removal is not implemented.

## Tensor invariants

Unless `unsafe=True`, parametrizations and non-`None` right-inverse results
must preserve shape and dtype. The source must be an `nn.Parameter`.
Modules without a `right_inverse` receive a method returning `None`, which
skips the pullback to the source parameter. Multiple parametrizations on one
parameter are applied in registration order; right inverses run in reverse.

This API uses explicit cache refreshes rather than the property-based access
provided by `torch.nn.utils.parametrize`.
"""

__all__ = [
    # Protocols
    "Surjection",
    "Parametrization",
    "Parametrized",
    "is_parametrization",
    "is_surjection",
    # Functions
    "assert_is_safe_parametrization",
    "get_parametrizations",
    "register_optimizer_hook",
    "update_parametrizations",
    # torch.nn.utils.parametrizations replacements
    "ParametrizationList",
    "is_parametrized",
    "register_parametrization",
    "remove_parametrizations",
    "cached",
]


import warnings
from abc import abstractmethod
from collections.abc import Iterable
from contextlib import AbstractContextManager, ContextDecorator
from types import TracebackType
from typing import (
    Any,
    Final,
    Literal,
    Optional,
    Protocol,
    Self,
    TypeIs,
    cast,
    get_protocol_members,
    overload,
    runtime_checkable,
)

import torch
from torch import Tensor, jit, nn
from torch.optim import Optimizer

from .containers import ModuleMapping, ModuleSequence


# TODO: Use IntersectionType
class Surjection[X, Y](Protocol):
    r"""Protocol for parametrization maps with an optional right inverse.

    A surjection $f:X→Y$ admits a right inverse $g:Y→X$ satisfying
    $f(g(y))=y$ for every $y∈Y$.

    For parametrizations, `right_inverse` may return `None` to skip updating the
    original parameter from the parametrized value. This permits maps without
    a right inverse at some parameter values. For example, the scaling map
    $x ↦ εx$ cannot be inverted onto nonzero outputs when $ε=0$.
    """

    @abstractmethod
    def __call__(self, x: X, /) -> Y: ...
    @abstractmethod
    def right_inverse(self, y: Y, /) -> X | None: ...


# TODO: Use IntersectionType
def is_surjection(obj: object, /) -> TypeIs[Surjection]:
    r"""Check whether an object is a module with a callable `right_inverse`.

    This checks the interface through dynamic attribute access. It does not
    verify surjectivity or the mathematical right-inverse identity.
    """
    return isinstance(obj, nn.Module) and callable(getattr(obj, "right_inverse", None))


@runtime_checkable
class Parametrization(Protocol):
    r"""Protocol for parametrizations."""

    @abstractmethod
    def __call__(self, arg: Tensor, /) -> Tensor:
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


@overload
def is_parametrization(obj: type, /) -> TypeIs[type[Parametrization]]: ...
@overload
def is_parametrization(obj: object, /) -> TypeIs[Parametrization]: ...
def is_parametrization(obj: object, /) -> bool:
    r"""Check whether all members of the parametrization protocol are callable.

    The check uses dynamic attribute access and accepts either an instance or
    a class. It checks the interface, without evaluating the parametrization.
    """
    return all(
        callable(getattr(obj, name, None))
        for name in get_protocol_members(Parametrization)
    )


# TODO: Use intersection type
class Parametrized(Protocol):
    r"""Protocol for `nn.Module` instances exposing a parametrization mapping."""

    parametrizations: ModuleMapping[ParametrizationList]


def is_parametrized(
    module: object, tensor_name: Optional[str] = None
) -> TypeIs[Parametrized]:
    r"""Check whether a module has registered parametrizations.

    If `tensor_name` is provided, check that specific parameter. Otherwise,
    require a nonempty parametrization mapping on the module itself.
    """
    return (
        isinstance(module, nn.Module)
        and ((parametrizations := get_parametrizations(module)) is not None)
        and (
            (tensor_name is None and len(parametrizations) > 0)
            or (tensor_name in parametrizations)
        )
    )


class _post_init_hook(type(Protocol)):  # pyrefly: ignore[invalid-inheritance]
    r"""Metaclass that calls `__post_init__` after instance initialization."""

    def __post_init__(cls, /) -> None:
        pass

    def __call__[T](cls: type[T], *args: Any, **kwargs: Any) -> T:
        instance = super().__call__(*args, **kwargs)  # ty: ignore[invalid-super-argument]
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
            namespace["__post_init__"] = _post_init_hook.__post_init__
            new = super().__new__(cls, name, bases, namespace, **kwargs)
        return new


# TODO: Use Intersection type (Surjection & nn.Module)
class ParametrizationList[
    S: Surjection,
](ModuleSequence[S], Parametrization, metaclass=_post_init_hook):  # type: ignore
    r"""Apply a chain of parametrizations to one source parameter.

    Args:
        modules: Modules applied in sequence. Modules without a `right_inverse`
            receive a method returning `None`, which skips the source pullback.
        original: Source `nn.Parameter`, stored as `original_parameter`.
        unsafe: Whether to skip shape and dtype checks on the transformations
            and their non-`None` right-inverse results.

    Note:
        The output is stored in `cached_parameter`. After an optimizer step or
        another source modification, refresh it with `update_parametrization`.
        Gradient accumulation marks the cache stale and fills it with NaNs.
    """

    original_parameter: nn.Parameter
    r"""PARAM: Original parameter from which the cached value is computed."""
    cached_parameter: Tensor
    r"""BUFFER: Holds cached version of the parametrized tensor."""
    is_stale: Tensor
    r"""BUFFER: Boolean scalar indicating whether the cached parameter is stale."""
    unsafe: Final[bool]
    r"""FLAG: Whether to skip shape and dtype checks."""

    def __init__(
        self,
        # TODO: use intersection type.
        modules: Iterable[nn.Module] = (),
        /,
        *,
        original: Tensor,
        unsafe: bool = False,
    ) -> None:
        super().__init__(modules)  # type: ignore
        if not isinstance(original, nn.Parameter):
            raise TypeError("tensor must be a nn.Parameter")

        # get the tensor to parametrizations
        self.register_parameter("original_parameter", original)
        self.register_buffer("cached_parameter", None)
        self.register_buffer("is_stale", torch.tensor(True), persistent=True)
        self.unsafe = unsafe

        # register the hook that makes the parametrized tensor stale after backward() call
        self.original_parameter.register_post_accumulate_grad_hook(self.set_stale)

    def __post_init__(self) -> None:
        if not self.unsafe:
            assert_is_safe_parametrization(self, self.original_parameter)

        # initialize the cache
        self.initialize_cache()

    def __setitem__(self, idx: int, module: S, /) -> None:  # type: ignore
        super().__setitem__(idx, _insert_right_inverse(module))  # type: ignore
        self.initialize_cache()

    def add_module(self, name: str, module: nn.Module | None) -> None:
        super().add_module(name, _insert_right_inverse(module))
        self.initialize_cache()

    def insert(self, index: int, module: nn.Module) -> None:
        super().insert(index, _insert_right_inverse(module))
        self.initialize_cache()

    def forward(self, x: Tensor) -> Tensor:
        for parametrization in self:
            x = parametrization(x)
        return x

    @jit.export
    def evaluate(self) -> Tensor:
        return self.forward(self.original_parameter)

    @jit.export
    def right_inverse(self, y: Tensor) -> Tensor | None:
        # NOTE: JIT does not support reversed().
        z: Tensor | None = y
        for parametrization in self[::-1]:
            if z is None:
                return z
            z = parametrization.right_inverse(z)
        return z

    @jit.export
    def initialize_cache(self) -> None:
        with torch.no_grad():
            # first, set the buffer, ensuring cached_parameter has
            # the correct tensor shape, dtype and device
            self.cached_parameter = self.evaluate().clone().detach_()
        self.cached_parameter.copy_(self.evaluate())

    @jit.export
    def set_stale(self, _: Optional[Tensor] = None) -> None:
        # poison the cached tensor
        self.cached_parameter.fill_(torch.nan)
        self.is_stale.fill_(True)

    @jit.export
    def set_fresh(self) -> None:
        # check that the tensor is no longer poisoned.
        if self.cached_parameter.isnan().all():
            raise ValueError("Parameter is poisoned, update parametrizations first.")
        self.is_stale.fill_(False)

    @jit.export
    def update_parametrization(self) -> None:
        r"""Refresh the cache and apply the right-inverse pullback when available.

        Call this after modifying the original parameter, including after an
        `optimizer.step()`. A `None` right-inverse result leaves the original
        parameter unchanged. The cache is then reconnected to autograd and
        marked fresh.

        See Also:
            - `update_parametrizations`: Refresh all parametrizations recursively.
            - `register_optimizer_hook`: Refresh caches after optimizer steps.
        """
        with torch.no_grad():
            # recompute the parametrization given the modified parameters
            self.cached_parameter.copy_(self.evaluate())

            # update the original parameters based on the new parametrization
            pullback = self.right_inverse(self.cached_parameter)
            if pullback is not None:
                self.original_parameter.copy_(pullback)

            # detach the cached tensors from the autograd engine
            self.cached_parameter.detach_()

        # re-enable the autograd engine
        self.cached_parameter.copy_(self.evaluate())
        self.set_fresh()


# TODO: use intersection type (Surjection & nn.Module)
@overload
def _insert_right_inverse(arg: None, /) -> None: ...
@overload
def _insert_right_inverse[M: nn.Module](arg: M, /) -> M: ...
def _insert_right_inverse(arg: nn.Module | None, /) -> nn.Module | None:
    if arg is None:
        return None

    if not isinstance(arg, nn.Module):
        raise TypeError(f"Expected a nn.Module, but got {type(arg)}!")

    if (right_inverse := getattr(arg, "right_inverse", None)) is not None:
        if not callable(right_inverse):
            raise TypeError(
                f"Given Module has a non-callable right_inverse attribute: {type(right_inverse)}"
            )

        # Ensure that right_inverse is jit-exported, otherwise we won't be able to call it
        # Note: we assume jit.export is idempotent.
        arg.__class__.right_inverse = jit.export(arg.__class__.right_inverse)  # type: ignore
        return arg

    # inject a trivial right_inverse method
    arg.__class__.right_inverse = jit.export(lambda _self, _tensor: None)  # type: ignore
    return arg


# region torch replacements  -----------------------------------------------------------
def register_parametrization(
    module: nn.Module,
    tensor_name: str,
    parametrization: nn.Module | None,
    *,
    unsafe: bool = False,
) -> None:
    r"""Register a parametrization, chaining multiple registrations via ParametrizationList."""
    parametrizations: ModuleMapping[ParametrizationList]

    match get_parametrizations(module):
        case None:
            module.register_module("parametrizations", ModuleMapping())
            assert isinstance(module.parametrizations, ModuleMapping)
            parametrizations = module.parametrizations

        case ModuleMapping() as parametrizations:  # pyright: ignore[reportAssignmentType]
            pass

        case nn.ModuleDict() as params:
            warnings.warn("Got nn.ModuleDict()", stacklevel=2)
            parametrizations = cast("ModuleMapping[ParametrizationList]", params)

        case other:
            raise TypeError(f"Expected a {ModuleMapping!s}, but got {type(other)}!")

    params_list: ParametrizationList

    match parametrizations.get(tensor_name):
        case None:
            # The tensor wasn't parametrized before.
            original = getattr(module, tensor_name)
            if not isinstance(original, nn.Parameter):
                raise TypeError(f"{tensor_name} is not a parameter!")

            params_list = ParametrizationList([], original=original, unsafe=unsafe)
            parametrizations[tensor_name] = params_list

            # remove the existing tensor
            delattr(module, tensor_name)

            # re-register as a buffer
            module.register_buffer(tensor_name, params_list.cached_parameter)

        case ParametrizationList() as params_list:
            # The tensor is already parametrized
            original = params_list.original_parameter

        case torch.nn.utils.parametrize.ParametrizationList():
            raise NotImplementedError(
                "This tensor appears to already be parametrized with"
                "torch.nn.utils.parametrize.ParametrizationList"
            )

        case other:
            raise TypeError(
                f"Expected a {ParametrizationList!s}, but got {type(other)}!"
            )

    if isinstance(parametrization, nn.Module):
        if not unsafe:
            assert_is_safe_parametrization(parametrization, original)

        # append the parametrization to the list
        params_list.append(parametrization)

    params_list.update_parametrization()

    # re-set the buffer
    setattr(module, tensor_name, params_list.cached_parameter)

    assert getattr(module, tensor_name) is params_list.cached_parameter


def assert_is_safe_parametrization(module: nn.Module, tensor: Tensor) -> None:
    r"""Check that a parametrization preserves tensor shape and dtype.

    A non-`None` right-inverse result must preserve these properties too.
    A right inverse raising `NotImplementedError` is accepted. Device equality
    and the mathematical right-inverse identity are not checked.
    """
    transform = _insert_right_inverse(module)
    assert is_surjection(transform)
    Y = tensor
    X = transform(Y)

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
        Z = transform.right_inverse(X)
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


def remove_parametrizations(
    module: nn.Module,
    tensor_name: str,
    leave_parametrized: bool = True,
) -> nn.Module:
    r"""Remove parametrizations from a module parameter.

    Raises:
        NotImplementedError: Parametrization removal is not implemented.
    """
    raise NotImplementedError


class cached(ContextDecorator, AbstractContextManager):
    r"""Refresh parametrization caches when leaving a context or decorated call.

    At least one module is required. Caches in the supplied modules and their
    descendants are refreshed on exit, including when the body raises an
    exception. The context does not suppress exceptions.
    """

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


# endregion torch replacements ---------------------------------------------------------


# region functions for parametrization -------------------------------------------------
def get_parametrizations(module: nn.Module, /) -> ModuleMapping[nn.Module] | None:
    r"""Return the (top level) parametrizations on a module, if present."""
    match ps := getattr(module, "parametrizations", None):
        case None:
            return None

        case ModuleMapping():
            return ps

        case nn.ModuleDict():
            return ModuleMapping(dict(ps.items()))

        case nn.Module() as ps:
            # Needed since JIT may replace ModuleDict with a plain Module.
            return ModuleMapping(dict(ps.named_children()))

        case _:
            raise TypeError(
                f"Expected parametrizations to be a ModuleMapping "
                f" or nn.Module, but got {type(ps)}!"
            )


def _supports_fallback_interface(parametrization: nn.Module, /) -> bool:
    return (
        isinstance(parametrization, nn.Module)
        and isinstance(getattr(parametrization, "cached_parameter", None), Tensor)
        and isinstance(getattr(parametrization, "original_parameter", None), Tensor)
        and isinstance(getattr(parametrization, "is_stale", None), Tensor)
    )


def _update_fallback(module: nn.Module, tensor_name: str, parametrization: nn.Module):
    if not _supports_fallback_interface(parametrization):
        raise TypeError(
            f"Parametrization {type(parametrization)} for {tensor_name} "
            f" does not support the minimal interface for fallback update."
        )

    # for type checker
    assert isinstance(parametrization.cached_parameter, Tensor)
    assert isinstance(parametrization.original_parameter, Tensor)
    assert isinstance(parametrization.is_stale, Tensor)

    with torch.no_grad():
        new_tensor = parametrization(parametrization.original_parameter)
        parametrization.cached_parameter.copy_(new_tensor)
        parametrization.cached_parameter.detach_()

    new_tensor = parametrization(parametrization.original_parameter)
    parametrization.cached_parameter.copy_(new_tensor)
    parametrization.is_stale.fill_(False)
    setattr(module, tensor_name, parametrization.cached_parameter)


def update_parametrizations(module: nn.Module, /) -> None:
    r"""Refresh parametrizations in a module and all its descendants."""
    for submodule in module.modules():
        if parametrizations := get_parametrizations(submodule):
            for tensor_name, parametrization in parametrizations.items():
                if callable(getattr(parametrization, "update_parametrization", None)):
                    # redundant, but needed for type checker
                    assert callable(parametrization.update_parametrization)

                    parametrization.update_parametrization()

                    # repair the buffer on the parent module
                    setattr(submodule, tensor_name, parametrization.cached_parameter)

                else:
                    _update_fallback(submodule, tensor_name, parametrization)


def register_optimizer_hook(optim: Optimizer, /, *modules: nn.Module) -> None:
    r"""Register an optimizer post-step hook that refreshes the supplied modules."""

    def hook(opt: Optimizer, /, *args: Any, **kwargs: Any) -> None:  # ruff: ignore[ARG001]
        r"""Hook to update the parametrization after each optimizer step."""
        for module in modules:
            update_parametrizations(module)

    optim.register_step_post_hook(hook)


# endregion functions for parametrization ----------------------------------------------
