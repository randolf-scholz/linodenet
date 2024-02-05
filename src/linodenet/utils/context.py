# type: ignore
"""Decorators and context managers for LinodeNet."""

__all__ = [
    "deprecated",
    "make_default_message",
    "make_wrapper",
    "timeout",
    "timer",
    "wrap_func",
    "wrap_type",
]

import gc
import logging
import signal
import sys
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager, ContextDecorator
from functools import wraps
from time import perf_counter_ns
from types import FrameType, TracebackType
from typing import ClassVar, Literal, ParamSpec, overload

from typing_extensions import Never, Self

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


class timeout(ContextDecorator, AbstractContextManager):
    """Context manager for timing out a block of code."""

    num_seconds: int
    timeout_occurred: bool

    def __init__(self, num_seconds: int) -> None:
        self.num_seconds = num_seconds
        self.timeout_occurred = False

    def _timeout_handler(self, signum: int, frame: None | FrameType) -> Never:
        self.timeout_occurred = True
        raise RuntimeError("Execution timed out")

    def __enter__(self) -> Self:
        # Set the signal handler for SIGALRM (alarm signal)
        signal.signal(signal.SIGALRM, self._timeout_handler)
        # Schedule the alarm to go off in num_seconds seconds
        signal.alarm(self.num_seconds)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Cancel the scheduled alarm
        signal.alarm(0)
        # Reset the signal handler to its default behavior
        signal.signal(signal.SIGALRM, signal.SIG_DFL)


class timer(ContextDecorator):
    """Context manager for timing a block of code."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}/{__qualname__}")

    start_time: int
    """Start time of the timer."""
    end_time: int
    """End time of the timer."""
    elapsed: float
    """Elapsed time of the timer in seconds."""

    def __enter__(self) -> Self:
        self.LOGGER.info("Flushing pending writes.")
        sys.stdout.flush()
        sys.stderr.flush()
        self.LOGGER.info("Disabling garbage collection.")
        gc.collect()
        gc.disable()
        self.LOGGER.info("Starting timer.")
        self.start_time = perf_counter_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self.end_time = perf_counter_ns()
        self.elapsed = (self.end_time - self.start_time) / 10**9
        self.LOGGER.info("Stopped timer.")
        gc.enable()
        self.LOGGER.info("Re-Enabled garbage collection.")
        return False


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
