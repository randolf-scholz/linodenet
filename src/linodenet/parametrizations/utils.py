r"""Utilities for parametrization."""

__all__ = ["resolve_matrix_parametrization"]

from typing import Optional

from torch import nn

from linodenet.utils import resolve_name


def resolve_matrix_parametrization(
    kernel_parametrization: Optional[str | nn.Module],
    /,
) -> nn.Module | None:
    match kernel_parametrization:
        case None:
            return None

        case nn.Module() as parametrization:
            return parametrization

        case str(key):
            assert __package__ is not None
            pkg = __import__(__package__)
            parametrization_cls = resolve_name(pkg.MATRIX_PARAMETRIZATIONS, key)

            try:
                parametrization = parametrization_cls()
            except Exception as exc:
                exc.add_note(
                    f"failed to initialize parametrization {parametrization_cls}"
                )
                raise

            assert isinstance(parametrization, nn.Module)
            return parametrization

        case _:
            raise TypeError(
                "kernel_parametrization must be a string, nn.Module, or None, "
                f"got {type(kernel_parametrization)!r}."
            )
