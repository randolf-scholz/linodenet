"""Utilities for testing."""

__all__ = [
    "assert_close",
    "timeout",
]

import signal
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager, ContextDecorator
from types import FrameType, TracebackType

import torch
from torch import Tensor
from typing_extensions import Never, Self

from linodenet.types import Nested


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


def assert_close(
    values: Nested[Tensor],
    reference: Nested[Tensor],
    /,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert that outputs and targets are close."""
    match values:
        case Tensor() as tensor:
            assert isinstance(reference, Tensor)
            assert torch.allclose(tensor, reference, rtol=rtol, atol=atol), (
                tensor,
                reference,
            )
        case Mapping() as mapping:
            assert isinstance(reference, Mapping)
            assert mapping.keys() == reference.keys()
            for key in mapping:
                x = mapping[key]
                y = reference[key]
                assert_close(x, y, rtol=rtol, atol=atol)
        case Iterable() as iterable:
            assert isinstance(reference, Iterable)
            for output, target in zip(iterable, reference, strict=True):
                assert_close(output, target, rtol=rtol, atol=atol)
        case _:
            raise TypeError(f"Unsupported type {type(values)} for `outputs`!")
