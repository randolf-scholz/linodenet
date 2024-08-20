r"""Context managers for use in decorators."""

__all__ = [
    # Classes (Context Managers)
    "timeout",
    "timer",
]

import gc
import logging
import signal
import sys
from contextlib import AbstractContextManager, ContextDecorator
from dataclasses import KW_ONLY, dataclass
from time import perf_counter_ns
from types import FrameType, TracebackType
from typing import ClassVar, Literal, Never, Self


class timer(ContextDecorator):
    r"""Context manager for timing a block of code."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}/{__qualname__}")

    start_time: int
    r"""Start time of the timer."""
    end_time: int
    r"""End time of the timer."""

    def __enter__(self) -> Self:
        r"""Disable garbage collection and start the timer."""
        # flush pending writes
        sys.stdout.flush()
        sys.stderr.flush()
        # disable garbage collection
        gc.collect()
        gc.disable()
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
        gc.enable()
        gc.collect()
        return False

    @property
    def elapsed_time(self) -> int:
        r"""Elapsed time in nanoseconds."""
        if start_time := getattr(self, "start_time", None) is None:
            raise RuntimeError("Timer has not been started!")
        if end_time := getattr(self, "end_time", None) is None:
            raise RuntimeError("Timer is still running!")
        return end_time - start_time

    @property
    def elapsed_seconds(self) -> float:
        r"""Elapsed time in seconds."""
        return self.elapsed_time / 1_000_000_000

    @property
    def value(self) -> str:
        r"""Formatted elapsed time."""
        hours, remainder = divmod(self.elapsed_time, 3_600_000_000_000)
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


@dataclass
class timeout(ContextDecorator, AbstractContextManager):
    r"""Context manager for timing out a block of code."""

    num_seconds: int

    _: KW_ONLY

    suppress: bool = False

    def __post_init__(self) -> None:
        self._exception = TimeoutError("Execution timed out.")

    def _timeout_handler(self, signum: int, frame: None | FrameType) -> Never:  # noqa: ARG002
        raise self._exception

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
        exc_tb: TracebackType | None,
        /,
    ) -> bool:
        # Cancel the scheduled alarm
        signal.alarm(0)
        # Reset the signal handler to its default behavior
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        if exc_type is self._exception:
            return self.suppress
        return False
