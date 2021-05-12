r"""
Initializations for the Linear ODE Network

Constants
---------

.. data:: INIT

    Dictionary containing all the available initializations

Functions
---------
"""

from .initializations import gaussian, symmetric, skew_symmetric, orthogonal, special_orthogonal

INITS = {
    'gaussian'           : gaussian,
    'symmetric'          : symmetric,
    'skew-symmetric'     : skew_symmetric,
    'orthogonal'         : orthogonal,
    'special-orthogonal' : special_orthogonal,
}

__all__ = ['INITS', 'gaussian', 'symmetric', 'skew_symmetric', 'orthogonal', 'special_orthogonal']
