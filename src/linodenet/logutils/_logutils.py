r"""Logging utilities for linodenet."""


# __all__ = []


# from typing import Protocol, TypeAlias, TypeVar, runtime_checkable
#
# Object = TypeVar("Object", bound=object, contravariant=True)
# FileWriter: TypeAlias = object
#
#
# def log_score(i: int, score: float, /, writer: FileWriter) -> None: pass
# def log_id(i: int, identity: int, /, writer: FileWriter) -> None: pass
# def log_items(i: int, items: list, /, writer: FileWriter, *, target: float, predict: float) -> None: pass
#
# y: Callback[float] = log_score  # ✘ Incompatible types in assignment
# z: Callback[int] = log_id  # ✘ Incompatible types in assignment
# w: Callback[list] = log_items  # ✘ Incompatible types in assignment
#
# from torch import nn
# from torch.utils.tensorboard import SummaryWriter
#
# from linodenet.models import LinODEnet
#
# Model = TypeVar("Model", bound=nn.Module, contravariant=True)

# @runtime_checkable
# class Callback(Protocol, Generic[Model]):
#
#     # @staticmethod
#     def __call__(
#         self,
#         i: int,
#         /,
#         writer: SummaryWriter,
#         model: Model,
#         *,
#         key: str = "",
#         name: str = "metrics",
#         prefix: str = "",
#         postfix: str = "",
#     ) -> None:
#         pass
#
#
# def log_linodenet_inverse_property(
#     i: int,
#     /,
#     writer: SummaryWriter,
#     model: LinODEnet,
#     *,
#     key: str = "",
#     name: str = "metrics",
#     prefix: str = "",
#     postfix: str = "",
# ) -> None:
#     r"""Log the inverse property of the LinodeNet.
#
#     Args:
#         i: The iteration of the training
#         writer: The tensorboard writer
#     """
#
#
# reveal_type(log_linodenet_inverse_property)
# y: type[nn.Module] = LinODEnet
# x: Callback = log_linodenet_inverse_property
# reveal_type(x)
