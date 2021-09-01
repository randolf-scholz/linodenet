r"""Provides utility functions.

linodenet.util
==============

.. data:: ACTIVATIONS

    A :class:`dict` containing string names as keys and corresponding :mod:`torch` functions.
"""

from linodenet.util.util import ACTIVATIONS, deep_dict_update, deep_keyval_update

__all__: list[str] = ["ACTIVATIONS", "deep_dict_update", "deep_keyval_update"]
