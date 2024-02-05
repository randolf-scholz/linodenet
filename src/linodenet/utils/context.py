"""Decorators and context managers for LinodeNet."""

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
from time import perf_counter_ns
from types import FrameType, TracebackType

from typing_extensions import ClassVar, Literal, Never, Self


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
