r"""Projections for the Linear ODE Networks.

Notes
-----
Contains projections in both modular and functional form.
  - See `~linodenet.projections.functional` for functional implementations.
  - See `~linodenet.projections.modular` for modular implementations.
"""


__all__ = [
    # Constants
    "PROJECTIONS",
    "FUNCTIONAL_PROJECTIONS",
    "MODULAR_PROJECTIONS",
    # Types
    "Projection",
    "ProjectionABC",
    # Sub-Modules
    "functional",
    "modular",
    # checks
    "is_banded",
    "is_diagonal",
    "is_masked",
    "is_normal",
    "is_orthogonal",
    "is_skew_symmetric",
    "is_symmetric",
    "is_traceless",
    # Functions
    "banded",
    "diagonal",
    "identity",
    "masked",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
    "traceless",
    # Classes
    "Banded",
    "Diagonal",
    "Identity",
    "Masked",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
    "Traceless",
]


from linodenet.projections import functional, modular
from linodenet.projections._projections import Projection, ProjectionABC
from linodenet.projections.functional import (
    banded,
    diagonal,
    identity,
    masked,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
    traceless,
)
from linodenet.projections.modular import (
    Banded,
    Diagonal,
    Identity,
    Masked,
    Normal,
    Orthogonal,
    SkewSymmetric,
    Symmetric,
    Traceless,
)
from linodenet.projections.testing import (
    is_banded,
    is_diagonal,
    is_masked,
    is_normal,
    is_orthogonal,
    is_skew_symmetric,
    is_symmetric,
    is_traceless,
)

FUNCTIONAL_PROJECTIONS: dict[str, Projection] = {
    "banded": banded,
    "diagonal": diagonal,
    "identity": identity,
    "masked": masked,
    "normal": normal,
    "orthogonal": orthogonal,
    "skew_symmetric": skew_symmetric,
    "symmetric": symmetric,
    "traceless": traceless,
}
r"""Dictionary of all available modular metrics."""

MODULAR_PROJECTIONS: dict[str, type[Projection]] = {
    "Banded": Banded,
    "Diagonal": Diagonal,
    "Identity": Identity,
    "Masked": Masked,
    "Normal": Normal,
    "Orthogonal": Orthogonal,
    "SkewSymmetric": SkewSymmetric,
    "Symmetric": Symmetric,
    "Traceless": Traceless,
}
r"""Dictionary of all available modular metrics."""

PROJECTIONS: dict[str, Projection | type[Projection]] = {
    **FUNCTIONAL_PROJECTIONS,
    **MODULAR_PROJECTIONS,
}
r"""Dictionary containing all available projections."""
