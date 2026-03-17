__all__ = ["timer"]


import gc
import logging
import signal
import sys
import threading
from contextlib import ContextDecorator
from time import perf_counter_ns
from types import FrameType, TracebackType
from typing import Any, ClassVar, Literal, Never, Self

from tqdm import tqdm

_TICK_INTERVAL = 200


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
