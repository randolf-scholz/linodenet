r"""Utility functions."""

__all__ = [
    # Types
    "NestedDict",
    "NestedMapping",
    # Classes
    "timer",
    # Functions
    "deep_dict_update",
    "flatten_dict",
    "get_module",
    "is_allcaps",
    "is_dunder",
    "is_private",
    "unflatten_dict",
]

import gc
import logging
import signal
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from contextlib import ContextDecorator
from copy import deepcopy
from time import perf_counter_ns
from types import FrameType, TracebackType
from typing import Any, ClassVar, Literal, Never, Self, cast, overload

from tqdm import tqdm


def is_allcaps(s: str, /) -> bool:
    r"""Check if a string is all caps."""
    return s.isidentifier() and s.isupper() and s.isalpha()


def is_dunder(s: str, /) -> bool:
    r"""Check if name is a dunder method."""
    return s.isidentifier() and s.startswith("__") and s.endswith("__")


def is_private(s: str, /) -> bool:
    r"""Check if name is a private method."""
    return s.isidentifier() and s.startswith("_") and not s.startswith("__")


def get_module(obj_ref: object, /) -> str:
    return obj_ref.__module__.rsplit(".", maxsplit=1)[-1]


def deep_dict_update(d: dict, new: Mapping, /, *, inplace: bool = False) -> dict:
    r"""Update nested dictionary recursively in-place with new dictionary.

    References:
        https://stackoverflow.com/a/30655448/9318372
    """
    if not inplace:
        d = deepcopy(d)

    for key, value in new.items():
        match value:
            # recurse on non-empty mapping
            case Mapping() as mapping if mapping:  # non-empty mapping
                subdict = d.get(key, {})
                d[key] = deep_dict_update(subdict, mapping, inplace=True)
            # update value for the given key
            case _:
                d[key] = value
    return d


type NestedDict[K, V] = dict[K, V | "NestedDict[K, V]"]
r"""Generic Type Alias for nested `dict`."""
type NestedMapping[K, V] = Mapping[K, V | "NestedMapping[K, V]"]
r"""Generic Type Alias for nested `Mapping`."""


@overload
def flatten_dict(
    d: NestedMapping[str, Any],
    /,
    *,
    join_fn: Callable[[Iterable[str]], str] = ...,
    split_fn: Callable[[str], Iterable[str]] = ...,
    recursive: bool | int = ...,
) -> dict[str, Any]: ...
@overload
def flatten_dict[K, K2](
    d: NestedMapping[K, Any],
    /,
    *,
    join_fn: Callable[[Iterable[K]], K2],
    split_fn: Callable[[K2], Iterable[K]],
    recursive: bool | int = ...,
) -> dict[K2, Any]: ...
def flatten_dict[K, K2](
    d: NestedMapping[K, Any],
    /,
    *,
    join_fn: Callable[[Iterable[K]], K2] = cast("Any", ".".join),  # noqa: B008
    split_fn: Callable[[K2], Iterable[K]] = cast("Any", lambda s: s.split(".")),  # noqa: B008
    recursive: bool | int = True,
) -> dict[K2, Any]:
    r"""Flatten dictionaries recursively.

    Args:
        d: dictionary to flatten
        recursive: whether to flatten recursively.
            If `recursive` is an integer, flattens that many levels.
        join_fn: function to join keys
            Defaults to ``'.'.join``, implicitly assuming that all keys are strings.
        split_fn: function to split keys
            Defaults to ``str.split('.')``, implicitly assuming that all keys are strings.

    Example: flattening with string keys.
        When ``join_fn`` and ``split_fn`` are not provided, they default to
        ``join_fn = ".".join`` and ``split_fn = lambda s: s.split(".")``,
        implicitly assuming that all keys are strings.
        This will combine string keys like ``"a"`` and ``"b"`` into ``"a.b"``.

        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {'a.b': 1, 'a.c': 2}

        >>> flatten_dict({"a": {"b": {"x": 2}, "c": 2}})
        {'a.b.x': 2, 'a.c': 2}

    Example: flattening with custom key functions.
        Using ``join_fn = tuple`` and ``split_fn = lambda s: s`` will combine
        keys like ``("a", "b")`` and ``("a", "c")`` into ``("a", "b", "c")``.
        This choice works for arbitrary key types.

        >>> flatten_dict({"a": {"b": 1, "c": 2}}, join_fn=tuple, split_fn=lambda x: x)
        {('a', 'b'): 1, ('a', 'c'): 2}

        >>> flatten_dict(
        ...     {"a": {"b": {"x": 2}, "c": 2}},
        ...     join_fn=tuple,
        ...     split_fn=lambda x: x,
        ... )
        {('a', 'b', 'x'): 2, ('a', 'c'): 2}

    Example: partial flattening with ``recursive``.
        >>> flatten_dict({"a": {"i": {"x": 0}, "b": {"y": 1}}})
        {'a.i.x': 0, 'a.b.y': 1}

        >>> flatten_dict(
        ...     {"a": {"i": {"x": 0}, "b": {"y": 1}}},
        ...     recursive=2,
        ... )
        {'a.i': {'x': 0}, 'a.b': {'y': 1}}

        >>> flatten_dict(
        ...     {"a": {"i": {"x": 0}, "b": {"y": 1}}},
        ...     recursive=1,
        ... )
        {'a': {'i': {'x': 0}, 'b': {'y': 1}}}
    """
    recursive = recursive if isinstance(recursive, bool) else recursive - 1
    result: dict[K2, Any] = {}
    for key, item in d.items():
        if recursive and isinstance(item, Mapping):
            for subkey, subitem in flatten_dict(
                item,
                recursive=recursive,
                join_fn=join_fn,
                split_fn=split_fn,
            ).items():
                new_key = join_fn((key, *split_fn(subkey)))
                result[new_key] = subitem
        else:
            new_key = join_fn((key,))
            result[new_key] = item
    return result


@overload
def unflatten_dict(
    d: Mapping[str, Any],
    /,
    *,
    join_fn: Callable[[Iterable[str]], str] = ...,
    split_fn: Callable[[str], Iterable[str]] = ...,
    recursive: bool | int = ...,
) -> NestedDict[str, Any]: ...
@overload
def unflatten_dict[K, K2](
    d: Mapping[K2, Any],
    /,
    *,
    join_fn: Callable[[Iterable[K]], K2],
    split_fn: Callable[[K2], Iterable[K]],
    recursive: bool | int = ...,
) -> NestedDict[K, Any]: ...
def unflatten_dict[K, K2](
    d: Mapping[K2, Any],
    /,
    *,
    recursive: bool | int = True,
    join_fn: Callable[[Iterable[K]], K2] = cast("Any", ".".join),  # noqa: B008
    split_fn: Callable[[K2], Iterable[K]] = cast("Any", lambda s: s.split(".")),  # noqa: B008
) -> NestedDict[K, Any]:
    r"""Unflatten dictionaries recursively.

    Example: Unflattening with string keys.
        When ``join_fn`` and ``split_fn`` are not provided, they default to
        ``join_fn = ".".join`` and ``split_fn = lambda s: s.split(".")``,
        implicitly assuming that all keys are strings.
        This will split up keys like ``"a.b"`` into ``{"a": {"b": ...}}``.
        >>> unflatten_dict({"a.b": 1, "a.c": 2})
        {'a': {'b': 1, 'c': 2}}

    Example: unflattening with custom join function.
        Using ``join_fn = tuple`` and ``split_fn = lambda s: s`` will split up
        keys like ``("a", "b", "c")`` into ``{"a": {"b": {"c": ...}}}``.
        >>> unflatten_dict(
        ...     {("a", 17): "foo", ("a", 18): "bar"},
        ...     join_fn=tuple,
        ...     split_fn=lambda x: x,
        ... )
        {'a': {17: 'foo', 18: 'bar'}}

    Example: partial unflattening with ``recursive``.
        >>> unflatten_dict({"a.b.c.d": 0, "a.x.y.z": 1})
        {'a': {'b': {'c': {'d': 0}}, 'x': {'y': {'z': 1}}}}

        >>> unflatten_dict({"a.b.c.d": 0, "a.x.y.z": 1}, recursive=2)
        {'a': {'b': {'c.d': 0}, 'x': {'y.z': 1}}}

        >>> unflatten_dict({"a.b.c.d": 0, "a.x.y.z": 1}, recursive=1)
        {'a': {'b.c.d': 0, 'x.y.z': 1}}
    """
    recursive = recursive if isinstance(recursive, bool) else recursive - 1
    result: dict[K, Any] = {}
    for key, item in d.items():
        outer_key, *inner_keys = split_fn(key)
        if inner_keys:
            result.setdefault(outer_key, {})
            if not isinstance(result[outer_key], dict):
                raise KeyError(f"Key conflict at {outer_key}! Cannot unflatten.")
            if recursive:
                result[outer_key] |= unflatten_dict(
                    {join_fn(inner_keys): item},
                    recursive=recursive,
                    split_fn=split_fn,
                    join_fn=join_fn,
                )
            else:
                result[outer_key] |= {join_fn(inner_keys): item}
        elif outer_key in result:
            raise KeyError(f"Key conflict at {outer_key}! Cannot unflatten.")
        else:
            result[outer_key] = item
    return result


_TICK_INTERVAL = 200
r"""Progress bar update interval in milliseconds."""


class timer(ContextDecorator):
    r"""Context manager for timing a block of code.

    Args:
        timeout: Timeout in seconds. ``None`` disables timeouts.
        progress_bar: If ``True``, enable a tqdm progress bar via :meth:`tick`.
    """

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}/{__qualname__}")

    start_time: int | None = None
    r"""Start time of the timer."""
    end_time: int | None = None
    r"""End time of the timer."""
    exc: TimeoutError | None = None
    r"""Timeout error raised by the timer, if any."""

    def __init__(
        self,
        timeout: int | None = None,
        *,
        disable_gc: bool = True,
        progress_bar: bool = False,
    ) -> None:
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive seconds or None.")
        if progress_bar and timeout is None:
            raise ValueError("progress_bar requires a timeout.")
        self.timeout = timeout
        self.disable_gc = disable_gc
        self.progress_bar = progress_bar
        self._alarm_handler: Any = None
        self._alarm_remaining: int = 0
        self._progress_bar: tqdm | None = None
        self._tick_interval: float = _TICK_INTERVAL / 1000
        self._tick_thread: threading.Thread | None = None
        self._tick_stop: threading.Event | None = None

    def _timeout_handler(self, signum: int, frame: FrameType | None) -> Never:  # noqa: ARG002
        self.exc = TimeoutError(f"Timed out after {self.timeout} seconds.")
        raise self.exc

    def __enter__(self) -> Self:
        r"""Disable garbage collection and start the timer."""
        self.exc = None
        self._progress_bar = None
        self._tick_stop = None
        self._tick_thread = None
        # flush pending writes
        sys.stdout.flush()
        sys.stderr.flush()

        # collect garbage
        gc.collect()

        # disable garbage collection
        if self.disable_gc:
            gc.disable()

        # set timeout if configured
        if self.timeout is not None:
            self._alarm_handler = signal.getsignal(signal.SIGALRM)
            self._alarm_remaining = signal.alarm(0)
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.timeout)

        # initialize progress bar if configured
        if self.progress_bar:
            self._progress_bar = tqdm(total=self.timeout, unit="s", leave=False)
            self._progress_bar.n = self._progress_bar.total
            self._progress_bar.set_description("Remaining time")
            self._progress_bar.refresh()
            self._tick_stop = threading.Event()
            self._tick_thread = threading.Thread(
                target=self.tick,
                name=f"{self.__class__.__qualname__}-tick",
                daemon=True,
            )
            self._tick_thread.start()

        # start timer
        self.start_time = perf_counter_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
        /,
    ) -> Literal[False]:
        r"""Stop the timer and re-enable garbage collection."""
        self.end_time = perf_counter_ns()

        if self.timeout is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._alarm_handler)
            if self._alarm_remaining:
                signal.alarm(self._alarm_remaining)

        if self.progress_bar:
            assert self._tick_stop is not None
            assert self._tick_thread is not None
            assert self._progress_bar is not None
            self._tick_stop.set()
            self._progress_bar.close()
            self._tick_thread.join(timeout=self._tick_interval)

        if exc_val is self.exc and self.exc is not None:
            # timeout occurred
            pass

        # re-enable garbage collection
        if self.disable_gc:
            gc.enable()

        # collect garbage
        gc.collect()
        return False

    @property
    def elapsed_nanos(self) -> int:
        r"""Elapsed time in nanoseconds."""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started!")
        if self.end_time is None:
            return perf_counter_ns() - self.start_time
        return self.end_time - self.start_time

    @property
    def elapsed_time(self) -> float:
        r"""Elapsed time in seconds (float)."""
        return self.elapsed_nanos / 1_000_000_000

    @property
    def elapsed_seconds(self) -> int:
        r"""Elapsed time in seconds (rounded)."""
        return self.elapsed_nanos // 1_000_000_000

    @property
    def remaining_time(self) -> float | None:
        r"""Remaining time in seconds (float)."""
        if self.timeout is None:
            return None
        return self.timeout - self.elapsed_time

    @property
    def remaining_seconds(self) -> int | None:
        r"""Remaining time in seconds (rounded)."""
        if self.timeout is None:
            return None
        return self.timeout - self.elapsed_seconds

    def tick(self) -> None:
        r"""Update the progress bar periodically until the timer stops."""
        assert self._tick_stop is not None
        assert self._progress_bar is not None
        while not self._tick_stop.wait(self._tick_interval):
            remaining = self.remaining_time
            assert remaining is not None
            self._progress_bar.n = round(remaining, 2)
            self._progress_bar.refresh()

    def value(self) -> str:
        r"""Formatted elapsed time."""
        return self._format_ns(self.elapsed_nanos)

    @staticmethod
    def _format_ns(ns: int, /) -> str:
        r"""Format nanoseconds into a human-readable string."""
        hours, remainder = divmod(ns, 3_600_000_000_000)
        minutes, remainder = divmod(remainder, 60_000_000_000)
        seconds, remainder = divmod(remainder, 1_000_000_000)
        milliseconds, remainder = divmod(remainder, 1_000_000)
        microseconds = remainder // 1_000

        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {seconds}s"
        if seconds:  # print 2 decimal places
            return f"{seconds}.{remainder // 10**7:02d}s"
        if milliseconds:  # print 2 decimal places
            return f"{milliseconds}.{remainder // 10**4:02d}ms"
        if microseconds:  # print 2 decimal places
            return f"{microseconds}.{remainder // 10}µs"
        return f"{remainder}ns"
