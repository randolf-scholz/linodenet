r"""Initializations for the Linear ODE Networks."""
import logging
from typing import Callable, Final

from torch import Tensor

from linodenet.init.initializations import (
    SizeLike,
    gaussian,
    orthogonal,
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
    "INITIALIZATIONS",
    "Initialization",
    "SizeLike",
    "gaussian",
    "symmetric",
    "skew_symmetric",
    "orthogonal",
    "special_orthogonal",
]

Initialization = Callable[[SizeLike], Tensor]
INITIALIZATIONS: Final[dict[str, Initialization]] = {
    "gaussian": gaussian,
    "symmetric": symmetric,
    "skew-symmetric": skew_symmetric,
    "orthogonal": orthogonal,
    "special-orthogonal": special_orthogonal,
}
