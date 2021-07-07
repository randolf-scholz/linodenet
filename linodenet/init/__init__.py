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
import sys

from .initializations import (
    gaussian,
    orthogonal,
    skew_symmetric,
    special_orthogonal,
    symmetric,
)

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)-s]\t[%(name)s]\t%(message)s",  # (%(filename)s:%(lineno)s)",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
    stream=sys.stdout,
)


INITS = {
    "gaussian": gaussian,
    "symmetric": symmetric,
    "skew-symmetric": skew_symmetric,
    "orthogonal": orthogonal,
    "special-orthogonal": special_orthogonal,
}

__all__ = [
    "INITS",
    "gaussian",
    "symmetric",
    "skew_symmetric",
    "orthogonal",
    "special_orthogonal",
]
