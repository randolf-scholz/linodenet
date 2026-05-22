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
- `parametrizations`: plug-in replacement for `torch.nn.utils.parametrizations`
    wraps a function Tensor -> Tensor into a parametrization.
- `cached`: (quasi) plug-in replacement for `torch.nn.utils.parametrizations.cached`
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
  `torch.nn.utils.parametrizations.register_parametrization`.
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
    - `parametrizations`: wraps a function Tensor -> Tensor into a parametrization.
    - `ParametrizationCache`: Base class for parametrization with multiple cached tensors.
- `ParametrizationDict`: Base class for complex parametrization with multiple parametrized and cached tensors.
"""

__all__ = [
    # Protocols
    "Surjection",
    "Parametrization",
    "Parametrized",
    # Functions
    "assert_is_safe_parametrization",
    "deepcopy_with_parametrizations",
    # "detach_caches",
    "get_parametrizations",
    "iter_parametrizations",
    "is_parametrization",
    "is_surjection",
    "register_optimizer_hook",
    # "update_caches",
    # "update_originals",
    "update_parametrizations",
    # torch.nn.utils.parametrizations replacements
    "ParametrizationList",
    "is_parametrized",
    "register_parametrization",
    "remove_parametrizations",
    "cached",
]


import copy
import warnings
from abc import abstractmethod
from collections.abc import Iterable, Iterator
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
    r"""A protocol for surjections.

    Surjections are maps that are onto, i.e., for every $y$ in $Y$, there exists
    an $x$ in $X$ such that $f(x) = y$. In particular, they admit a right inverse,
    i.e., a map $g: Y -> X$ such that $f(g(y)) = y$ for all $y$ in $Y$.

    In the context of parametrizations, we allow surjections to return `None` when
    calling right_inverse. This is to signal that the right-inverse should not be used,
    and the parametrized tensor should not be updated with the pullback.

    For example, with the ReZero transform $x ↦ ε⋅x$, since the learnable scalar is initialized with $ε=0$,
    a `right_inverse` is not not well-defined at initialization.
    """

    @abstractmethod
    def __call__(self, x: X, /) -> Y: ...
    @abstractmethod
    def right_inverse(self, y: Y, /) -> X | None: ...


# TODO: Use IntersectionType
def is_surjection(obj: object, /) -> TypeIs[Surjection]:
    r"""Check if the object is a Surjection.

    This method is needed because standard isinstance checks do not work with jit.ScriptModule.
    This is because Protocol used getattr_static, rather than proper getattr/hasattr.
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

    @abstractmethod
    def get_cached_tensor(self) -> Tensor:
        r"""Get the cached tensor."""
        ...


@overload
def is_parametrization(obj: type, /) -> TypeIs[type[Parametrization]]: ...
@overload
def is_parametrization(obj: object, /) -> TypeIs[Parametrization]: ...
def is_parametrization(obj: object, /) -> bool:
    r"""Check if the object is a Parametrization.

    This method is needed because standard isinstance checks do not work with jit.ScriptModule.
    This is because Protocol used getattr_static, rather than proper getattr/hasattr.
    """
    return all(
        callable(getattr(obj, name, None))
        for name in get_protocol_members(Parametrization)
    )


# TODO: Use intersection type
class Parametrized(Protocol):
    r"""Protocol for parametrized modules.

    This should only apply to nn.Modules.
    """

    parametrizations: ModuleMapping[ParametrizationList]


def is_parametrized(
    module: object, tensor_name: Optional[str] = None
) -> TypeIs[Parametrized]:
    r"""Return True if the module has any parametrizations."""
    return (
        isinstance(module, nn.Module)
        and ((parametrizations := get_parametrizations(module)) is not None)
        and (
            (tensor_name is None and len(parametrizations) > 0)
            or (tensor_name in parametrizations)
        )
    )


# region base classes ------------------------------------------------------------------
class _post_init_hook(type(Protocol)):  # pyrefly: ignore[invalid-inheritance]
    r"""Metclass that adds a ``__post_init__`` hook to class initialization."""

    def __post_init__(cls, /) -> None:
        pass

    def __call__[T](cls: type[T], *args: Any, **kwargs: Any) -> T:
        instance = super().__call__(*args, **kwargs)
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
](ModuleSequence[S], Parametrization, metaclass=_post_init_hook):  # type: ignore[bad-specialization]
    r"""Applies multiple parametrizations to the same tensor in sequence.

    Args:
        modules (Iterable[S | nn.Module]): An iterable of parametrization modules to apply in
            sequence. Each entry should either implement the `Surjection` protocol (i.e. provide
            a callable transform and a `right_inverse`) or be a plain `nn.Module`, in which case
            it will be wrapped by `WithoutRightInverse` so that a `right_inverse` returning
            `None` is used.
        original (Tensor): The original tensor (typically an `nn.Parameter`) that is being
            parametrized. This tensor is stored as `original_parameter` and is the source of
            truth for updating the cached parametrized value. Its dtype and shape must be
            compatible with the parametrizations unless `unsafe=True` is set.
        unsafe (bool): If True, safety checks that ensure the parametrization preserves dtype
            and shape (and that the `right_inverse` respects these invariants) are skipped.
            Use with caution; enabling `unsafe` can lead to silently incorrect parametrizations.
    """

    original_parameter: nn.Parameter
    r"""PARAM: Holds parametrized tensors."""
    cached_parameter: Tensor
    r"""BUFFER: Holds cached version of the parametrized tensor."""
    is_stale: Tensor
    r"""BUFFER: Boolean scalar indicating whether the cached parameter is stale."""
    unsafe: Final[bool]
    r"""FLAG: Whether to perform safety checks (tests if parametrization changes dtype/device/shape)."""

    def __init__(
        self,
        # TODO: use intersection type.
        modules: Iterable[nn.Module] = (),
        /,
        *,
        original: Tensor,
        unsafe: bool = False,
    ) -> None:
        super().__init__(modules)  # type: ignore[arg-type]
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
        self.cached_parameter = self.apply_parametrization().clone().detach_()
        self.update_cache()

    def __setitem__(self, idx: int, module: S, /) -> None:  # type: ignore[override]
        super().__setitem__(idx, _insert_right_inverse(module))  # type: ignore[arg-type]
        self.cached_parameter = self.apply_parametrization().clone().detach_()

    def add_module(self, name: str, module: nn.Module | None) -> None:
        super().add_module(name, _insert_right_inverse(module))
        self.cached_parameter = self.apply_parametrization().clone().detach_()

    def insert(self, index: int, module: nn.Module) -> None:
        super().insert(index, _insert_right_inverse(module))
        self.cached_parameter = self.apply_parametrization().clone().detach_()

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        for parametrization in self:
            x = parametrization(x)
        return x

    @jit.export
    def right_inverse(self, y: Tensor) -> Tensor | None:
        # NOTE: JIT does not support reversed().
        z: Tensor | None = y
        for parametrization in self[::-1]:
            if z is None:
                return z
            z = parametrization.right_inverse(z)
        return z

    # implement protocol methods:
    @jit.export
    def apply_parametrization(self) -> Tensor:
        r"""Apply the parametrization to the weight matrix."""
        return self.forward(self.original_parameter)

    @jit.export
    def get_original_tensor(self) -> Tensor:
        r"""Get the original tensor from the cached tensor."""
        return self.original_parameter

    @jit.export
    def get_cached_tensor(self) -> Tensor:
        r"""Get the cached tensor from the original tensor."""
        return self.cached_parameter

    @jit.export
    def set_stale(self, _: Optional[Tensor] = None) -> None:
        self.is_stale.fill_(True)
        # poison the cached tensor
        self.cached_parameter.fill_(torch.nan)

    @jit.export
    def set_fresh(self) -> None:
        self.is_stale.fill_(False)
        # check that the tensor is no longer poisoned.
        assert not self.cached_parameter.isnan().all()

    @jit.export
    def update_cache(self) -> None:
        r"""Update the cached tensors by recomputing the parametrization using the original tensors.

        Note:
            This method should use inplace `copy_` operations to update the cached tensors.
        """
        new_tensor = self.apply_parametrization()
        self.cached_parameter.copy_(new_tensor)

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
            pullback = self.right_inverse(self.cached_parameter)
            if pullback is not None:
                self.original_parameter.copy_(pullback)

            # detach the cached tensors from the autograd engine
            self.cached_parameter.detach_()

        # re-enable the autograd engine
        self.update_cache()
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
        arg.__class__.right_inverse = jit.export(arg.__class__.right_inverse)  # type: ignore[attribute]
        return arg

    # inject a trivial right_inverse method
    arg.__class__.right_inverse = jit.export(lambda _self, _tensor: None)  # type: ignore[attribute]
    return arg


# endregion base classes ---------------------------------------------------------------


# region torch replacements  -----------------------------------------------------------
def register_parametrization(
    module: nn.Module,
    tensor_name: str,
    parametrization: nn.Module,
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

        case nn.ModuleDict() as _parametrizations:
            warnings.warn("Got nn.ModuleDict()", stacklevel=2)
            parametrizations = cast(
                "ModuleMapping[ParametrizationList]", _parametrizations
            )

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
            module.register_buffer(tensor_name, params_list.get_cached_tensor())

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

    # wrap the parametrization if needed
    # parametrization = parametrized(original, parametrization, unsafe=unsafe)
    assert isinstance(parametrization, nn.Module)
    # assert is_parametrization(parametrization)
    if not unsafe:
        assert_is_safe_parametrization(parametrization, original)

    # append the parametrization to the list
    params_list.append(parametrization)
    params_list.update_parametrization()

    # re-set the buffer
    setattr(module, tensor_name, params_list.cached_parameter)

    assert getattr(module, tensor_name) is params_list.cached_parameter


def assert_is_safe_parametrization(module: nn.Module, tensor: Tensor) -> None:
    r"""Check if the parametrization is safe to apply to the tensor."""
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


def get_parametrizations(module: nn.Module, /) -> ModuleMapping[nn.Module] | None:
    r"""Return all parametrizations in a module, if present."""
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
            raise TypeError(f"Expected a ModuleMapping, but got {type(ps)}!")


def remove_parametrizations(
    module: nn.Module,
    tensor_name: str,
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


# endregion torch replacements ---------------------------------------------------------


# region functions for parametrization -------------------------------------------------
def iter_parametrizations(module: nn.Module, /) -> Iterator[Parametrization]:
    r"""Yields all parametrizations in a module."""
    for child in module.children():
        if is_parametrization(child):
            yield child
        else:
            yield from iter_parametrizations(child)


def _heal_parametrization_connections(module: nn.Module, /) -> None:
    r"""Rebind parametrized module buffers to the wrapper caches.

    This restores aliasing after operations such as ``module.to(...)`` that rewrite
    buffers in each submodule independently.
    """
    for submodule in module.modules():
        match getattr(submodule, "parametrizations", None):
            case None:
                continue

            case nn.Module() as ps:
                if not isinstance(ps, ModuleMapping):
                    ps = ModuleMapping(dict(ps.named_children()))

                for tensor_name, parametrization in ps.items():
                    assert is_parametrization(parametrization)
                    setattr(submodule, tensor_name, parametrization.get_cached_tensor())

            case _:
                raise TypeError("Expected parametrizations to be a ModuleMapping")


def update_parametrizations(module: nn.Module, /) -> None:
    r"""Update all parametrizations in a module."""
    for parametrization in iter_parametrizations(module):
        parametrization.update_parametrization()
    _heal_parametrization_connections(module)


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
