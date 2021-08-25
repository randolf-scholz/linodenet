r"""Initializations for the Linear ODE Networks.

linodenet.init
==============

Constants
---------

.. data:: INIT

    Dictionary containing all the available initializations

Functions
---------
"""
import logging
from typing import Callable, Final

from torch import Tensor

from .initializations import (
    gaussian,
    orthogonal,
    SizeLike,
    skew_symmetric,
    special_orthogonal,
    symmetric,
)

# logging.basicConfig(
#     format="[%(asctime)s] [%(levelname)-s]\t[%(name)s]\t%(message)s (%(filename)s:%(lineno)s)",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     level=logging.DEBUG,
#     stream=sys.stdout,
# )

logger = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "INITS",
    "SizeLike",
    "gaussian",
    "symmetric",
    "skew_symmetric",
    "orthogonal",
    "special_orthogonal",
]


INITS: Final[dict[str, Callable[[SizeLike], Tensor]]] = {
    "gaussian": gaussian,
    "symmetric": symmetric,
    "skew-symmetric": skew_symmetric,
    "orthogonal": orthogonal,
    "special-orthogonal": special_orthogonal,
}
