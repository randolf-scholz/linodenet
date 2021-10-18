r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if `x~𝓝(0,1)`, then `Ax~𝓝(0,1)` as well.

Notes
-----
Contains initializations in modular form.
  - See :mod:`~.functional` for functional implementations.
"""

import logging

__logger__ = logging.getLogger(__name__)
