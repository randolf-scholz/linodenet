# type: ignore
"""Deprecated classes and functions."""

__all__ = [
    # Functions (Decorators)
    "deprecated",
    "make_default_message",
    "make_wrapper",
    "wrap_func",
    "wrap_type",
]


import warnings
from collections.abc import Callable
from functools import wraps

from typing_extensions import ParamSpec, overload

from linodenet.types import R, T

P = ParamSpec("P")


def make_default_message(obj: type | Callable, /) -> str:
    """Create a default deprecation message for an object."""
    match obj:
        case type() as cls:
            return f"Class {cls.__name__!r} is deprecated"
        case Callable() as func:
            assert hasattr(func, "__name__")
            if hasattr(func, "__self__"):
                return f"Method {func.__name__!r} of {func.__self__!r} is deprecated"

            return f"Function {func.__name__} is deprecated"
        case _:
            raise TypeError(f"Cannot create default message for {obj!r}")


def make_wrapper(
    obj: T,
    msg: str,
    *,
    category: type[Warning] = DeprecationWarning,
    stacklevel: int = 1,
) -> T:
    """Create a wrapper for a deprecated object."""
    msg = make_default_message(obj) if msg is NotImplemented else msg
    match obj, category:
        case _, None:
            obj.__deprecated__ = msg
            wrapped = obj
        case type(), _:
            wrapped = wrap_type(obj, msg, category=category, stacklevel=stacklevel)
        case Callable(), _:
            wrapped = wrap_func(obj, msg, category=category, stacklevel=stacklevel)
        case _:
            raise TypeError(
                "@deprecated decorator with non-None category must be applied to "
                f"a class or callable, not {obj!r}"
            )
    return wrapped


@overload
def deprecated() -> Callable[[T], T]: ...
@overload
def deprecated(msg: str, /) -> Callable[[T], T]: ...
@overload
def deprecated(decorated: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def deprecated(decorated: Callable[P, R], msg: str, /) -> Callable[P, R]: ...
@overload
def deprecated(decorated: type[T], /) -> type[T]: ...
@overload
def deprecated(decorated: type[T], msg: str, /) -> type[T]: ...
def deprecated(
    decorated=NotImplemented,
    msg=NotImplemented,
    /,
    *,
    category=DeprecationWarning,
    stacklevel=1,
):
    """Indicate that a class, function or overload is deprecated.

    See PEP 702 for details.
    """
    if isinstance(decorated, str):
        # used as deprecated("message") -> shift arguments
        assert msg is NotImplemented
        msg = decorated
        decorated = NotImplemented

    if decorated is NotImplemented:
        # used with brackets -> decorator factory
        def decorator(obj: T, /) -> T:
            """Decorate a function, method or class."""
            return make_wrapper(obj, msg, category=category, stacklevel=stacklevel)

        return decorator

    msg = make_default_message(decorated)
    return make_wrapper(decorated, msg, category=category, stacklevel=stacklevel)


def wrap_func(
    func: Callable[P, R],
    msg: str,
    *,
    category: type[Warning] = DeprecationWarning,
    stacklevel: int = 1,
) -> Callable[P, R]:
    """Wrap a function with a deprecation warning."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(msg, category=category, stacklevel=stacklevel + 1)
        return func(*args, **kwargs)

    func.__deprecated__ = wrapper.__deprecated__ = msg
    return wrapper


def wrap_type(
    klass: type[T],
    msg: str,
    *,
    category: type[Warning] = DeprecationWarning,
    stacklevel: int = 1,
) -> type[T]:
    """Wrap a class with a deprecation warning."""
    original_new = klass.__new__
    has_init = klass.__init__ is not object.__init__

    @wraps(original_new)
    def __new__(T, *args, **kwargs):
        warnings.warn(msg, category=category, stacklevel=stacklevel + 1)
        if original_new is not object.__new__:
            return original_new(T, *args, **kwargs)
        # Mirrors a similar check in object.__new__.
        if not has_init and (args or kwargs):
            raise TypeError(f"{T.__name__}() takes no arguments")
        return original_new(T)

    klass.__new__ = staticmethod(__new__)
    klass.__deprecated__ = __new__.__deprecated__ = msg
    return klass
