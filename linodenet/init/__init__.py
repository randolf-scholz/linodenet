r"""
Initializations
===============

Initializations for the Linear ODE Network

Constants
---------

.. data:: INIT

    Dictionary containing all the available initializations

Functions
---------
"""
import sys
import logging
from .initializations import gaussian, symmetric, skew_symmetric, orthogonal, special_orthogonal

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)-s]\t[%(name)s]\t%(message)s",  # (%(filename)s:%(lineno)s)",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
    stream=sys.stdout)


INITS = {
    'gaussian'           : gaussian,
    'symmetric'          : symmetric,
    'skew-symmetric'     : skew_symmetric,
    'orthogonal'         : orthogonal,
    'special-orthogonal' : special_orthogonal,
}

__all__ = ['INITS', 'gaussian', 'symmetric', 'skew_symmetric', 'orthogonal', 'special_orthogonal']
