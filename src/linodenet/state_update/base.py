r"""Different state-update models to be used in conjunction with LinODENet.

A state update is a map of the form $x' = F(y, x)$.
The square case `input_size == hidden_size` is common, but not universal.

A state updater takes two positional inputs:

- An input tensor y: the current measurement of the system
- An input tensor x: the current estimation of the state of the system

Sometimes, we have a third input, so called covariates $u$.
There are two types of covariates:

- Control inputs: These are external variables that influence the system
- Exogenous inputs: Sometimes, there are two coupled systems, and we have access to
  measurements / predictions of the other system (example: weather forecast).
  In this case we can treat these variables as part of the state.

These are external variables that influence the system,
but are not part of the state.

Example:
    The linear state space system is given by the equations (without noise):

    .. math::
        ẋ(t) &= A(t)x(t) + B(t)u(t) \\
        y(t) &= C(t)x(t) + D(t)u(t)

    Here $u$ is the control input.
"""

__all__ = [
    # Protocols
    "AbstractStateUpdate",
    "VectorStateUpdate",
    "SparseVectorStateUpdate",
    # classes
    "CellSequence",
    "ResidualCellSequence",
    "ResidualCell",
    "MissingValueCell",
    # functions
    "is_vector_state_updater",
    "is_idempotent_update",
]

from collections.abc import Iterable
from typing import Final, Protocol, TypeIs, cast

import torch
from torch import Tensor, nn

from linodenet.nn import ModuleSequence
from linodenet.nn.rezero import resolve_gate
from linodenet.testing.utils import get_device
from signatures import signature

from .imputation import (
    ConstantImputer,
    ImputationStrategy,
    ImputerProtocol,
    LastValueImputer,
    LearnableImputer,
    LinearImputer,
    ZeroImputer,
)


class AbstractStateUpdate[X, Y](Protocol):
    r"""Abstract protocol for state-update callbacks.

    .. math::  x' = F(y, x)
    """

    def __call__(self, y: Y, x: X, /) -> X: ...


class VectorStateUpdate(AbstractStateUpdate[Tensor, Tensor], Protocol):
    r"""Protocol for vector-valued state updaters.

    .. math::  x' = F(y, x)
    """

    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(self, /, input_size: int, hidden_size: int) -> None:
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

    @signature("[(..., d), (..., h)] -> (..., h)")
    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor: ...


class SparseVectorStateUpdate(VectorStateUpdate, Protocol):
    r"""Protocol for vector-valued state updaters that can handle missing values."""

    @signature("[(..., d), (..., h), (..., d)?] -> (..., h)")
    def __call__(
        self, y: Tensor, x: Tensor, /, *, mask: Tensor | None = None
    ) -> Tensor:
        r"""Update the state vector using an optional observation mask.

        Args:
            y: Observation vector.
            x: Current state estimate.
            mask: Mask indicating which coordinates of `y` are observed. If
                `None`, observed coordinates are inferred from finite values in `y`.
        """
        ...


def is_vector_state_updater(arg: object, /) -> TypeIs[VectorStateUpdate]:
    r"""Check whether an object is a state updater."""
    return (
        callable(arg)
        and isinstance(getattr(arg, "input_size", None), int)
        and isinstance(getattr(arg, "hidden_size", None), int)
    )


# TODO: use Intersection type
class CellSequence[C: VectorStateUpdate](ModuleSequence[C], VectorStateUpdate):  # type: ignore[bad-specialization]
    r"""Apply multiple state updaters sequentially.

    .. math:: xₖ₊₁ = Fₖ(y, xₖ)
    """

    def __init__(self, cells: Iterable[C] = (), /) -> None:
        cells = list(cells)

        if not cells:
            raise ValueError("At least one module must be given!")

        input_size = cells[0].input_size
        hidden_size = cells[0].hidden_size
        for module in cells:
            if module.input_size != input_size:
                raise ValueError(
                    "All modules must have the same input_size!"
                    f"Expected {input_size}, but {module=} has {module.input_size}"
                )
            if module.hidden_size != hidden_size:
                raise ValueError(
                    "All modules must have the same hidden_size!"
                    f"Expected {hidden_size}, but {module=} has {module.hidden_size}"
                )

        # ⚠️ multiple inheritance ⚠️
        # due to how nn.Module.__init__ works, it should only be ever called once
        # because it will overwrite internal state otherwise.
        # Therefore, we need to carefully manually reproduce the __init__ logic here.
        assert not hasattr(self, "_modules"), f"Module already initialized: {self}"
        super().__init__(cells)
        VectorStateUpdate.__init__(self, input_size, hidden_size)

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor, /) -> Tensor:
        for cell in self:
            x = cell(y, x)
        return x


class ResidualCellSequence[C: VectorStateUpdate](CellSequence[C]):
    r"""Sequential state updater with residual connections.

    .. math:: xₖ₊₁ = xₖ - αₖ⋅Fₖ(y, xₖ)

    Args:
        cells: An iterable of state updater modules to be applied sequentially.
        use_rezero: Whether to use rezero (default: True)

    A regular ResNet is obtained by setting all αₖ=1.0 and making them non-learnable.
    """

    alpha: Tensor
    r"""PARAM: The residual scaling factors αₖ."""
    use_rezero: Tensor
    r"""Whether to use rezero"""

    def __init__(
        self,
        cells: Iterable[C] = (),
        /,
        *,
        use_rezero: bool = True,
    ) -> None:
        super().__init__(cells)
        self.alpha = nn.Parameter(
            torch.zeros(len(self)) if use_rezero else torch.ones(len(self)),
            requires_grad=use_rezero,
        )

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        for alpha, cell in zip(self.alpha, self, strict=True):
            x = x.addcmul(alpha, cell(y, x), value=-1.0)  # xₖ₊₁ <- xₖ - αₖfₖ(y, xₖ)
        return x


class ResidualCell[C: VectorStateUpdate](nn.Module, VectorStateUpdate):
    r"""Residual wrapper for state updaters.

    .. math:: x' = x - ρ(F(y, x))

    where $F$ is a state updater and $ρ$ is an optional gate.
    """

    cell: C
    gate: nn.Module

    def __init__(self, cell: C, gate: str | nn.Module | None = None) -> None:
        super().__init__()
        VectorStateUpdate.__init__(
            self, input_size=cell.input_size, hidden_size=cell.hidden_size
        )
        self.cell = cell
        self.gate = resolve_gate(gate)

    @signature("[(..., d), (..., h)] -> (..., h)")
    def forward(self, y: Tensor, x: Tensor, /) -> Tensor:
        return x - self.gate(self.cell(y, x))


class MissingValueCell[F: VectorStateUpdate](nn.Module, SparseVectorStateUpdate):
    r"""Wraps an existing state updater $F$ so that it can handle missing values.

    .. math:: x' &= F(u，x)   &   u = impute(y, x; mask=m)

    where $u$ is an imputed value that is free of missing values.
    There are several available imputation strategies:

    0. "default": uses "decoder", if available, and "zero" otherwise.
    1. "zero": Replace missing values with zeros.
    2. "constant": Replace missing values with a constant value.
    3. "last": Replace missing values with the last observed value. (initialized with zero)
    4. "decoder": Replace missing values with the output of the decoder: $s = h(x)$.
    5. Tensor: replaces missing values with a fixed tensor. (for example, the mean of the data)

    Here `m=True` marks the coordinates that should be imputed. If `m=None`,
    the imputation mask defaults to `y.isnan()`.

    Optionally, the imputation mask can be concatenated to the input.

    .. math:: u' = concat([impute(y, x; mask=m)，m])
    """

    state_updater: F
    imputer: ImputerProtocol

    # CONSTANTS
    concat_mask: Final[bool]
    r"""CONST: Whether to concatenate the mask to the input or not."""
    imputation_strategy: Final[str]
    r"""CONST: The strategy to use for imputation."""
    # BUFFERS
    mask: Tensor
    r"""BUFFER: The mask tensor (true where values were imputed)."""
    imputed: Tensor
    r"""BUFFER: The most recent imputed value."""

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "filter": self.state_updater,
            "concat_mask": self.concat_mask,
            "imputation": self.imputation,
        }

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        state_updater: F,
        concat_mask: bool = True,
        imputation: str | float | Tensor | nn.Module = "zero",
    ) -> None:
        super().__init__()
        VectorStateUpdate.__init__(self, input_size=input_size, hidden_size=hidden_size)
        self.imputation = imputation
        self.concat_mask = bool(concat_mask)

        expected_input_size = self.input_size * (1 + self.concat_mask)
        if getattr(state_updater, "input_size", None) != expected_input_size:
            raise ValueError(
                "MissingValueCell requires a filter with "
                f"input_size={expected_input_size}, got "
                f"{getattr(state_updater, 'input_size', None)!r}."
            )
        if getattr(state_updater, "hidden_size", None) != self.hidden_size:
            raise ValueError(
                "MissingValueCell requires a filter with "
                f"hidden_size={self.hidden_size}, got "
                f"{getattr(state_updater, 'hidden_size', None)!r}."
            )
        self.state_updater = state_updater

        # initialize imputation strategy
        # imputation_strategy: ImputationStrategy
        imputer: ImputerProtocol
        match imputation:
            case "zero":
                imputation_strategy = ImputationStrategy.ZERO
                imputer = ZeroImputer()
            case "last":
                imputation_strategy = ImputationStrategy.LAST
                imputer = LastValueImputer()
            case "learnable":
                imputation_strategy = ImputationStrategy.LEARNABLE
                imputer = LearnableImputer(input_size)
            case "linear":
                imputation_strategy = ImputationStrategy.LINEAR
                imputer = LinearImputer(
                    input_size=self.input_size,
                    hidden_size=hidden_size,
                )
            case (Tensor() | float()) as value:
                imputation_strategy = ImputationStrategy.CONSTANT
                imputer = ConstantImputer(value)
            case nn.Module() as module:
                imputation_strategy = "other"
                imputer = cast("ImputerProtocol", module)
            case _:
                raise ValueError(f"Unknown imputation strategy: {imputation}")

        # FIXME: https://github.com/python/mypy/issues/10736
        #   Need to unconditionally assign Final due to mypy bug
        self.imputation_strategy = imputation_strategy
        self.imputer = imputer

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor, /, *, mask: Tensor | None = None) -> Tensor:
        # impute missing values and store the imputation mask
        self.mask = y.isnan() if mask is None else mask
        self.imputed = self.imputer(y, x, mask=self.mask)
        u = self.imputed

        if self.concat_mask:
            u = torch.cat([u, self.mask], dim=-1)

        return self.state_updater(u, x)


def is_idempotent_update(
    update: SparseVectorStateUpdate,
    /,
    *,
    batch_shape: tuple[int, ...] = (8,),
    check_sparse_observations: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    r"""Check the square-update idempotency condition on random data.

    This checks the direct-observation criterion $x=y ⟹ F(y, x)=x$ for a
    square state updater.

    Args:
        update: The state updater to test. Must satisfy
            `update.input_size == update.hidden_size`.
        batch_shape: Optional leading batch dimensions for the random test data.
        check_sparse_observations: Whether to also test the sparse-observation
            case where only a random subset of coordinates is observed.
        rtol: Relative tolerance for the equality check.
        atol: Absolute tolerance for the equality check.

    Returns:
        `True` if the update is idempotent on the sampled data, else `False`.

    Raises:
        ValueError: If the updater is not square.
    """
    if update.input_size != update.hidden_size:
        raise ValueError(
            "Idempotency requires a square state updater with "
            f"{update.input_size=} and {update.hidden_size=}."
        )

    device = get_device(update)
    x = torch.randn(*batch_shape, update.hidden_size, device=device)
    with torch.no_grad():
        # check that y=x ⟹ F(y,x)=x
        y = x.clone()
        if not torch.allclose(update(y, x), x, rtol=rtol, atol=atol):
            return False

        if not check_sparse_observations:
            return True

        # check that y=x ⟹ F([m ? y : NaN], x)=x for a random binary mask m
        mask = torch.rand(*batch_shape, update.input_size, device=device) > 0.5
        y_obs = torch.where(mask, y, torch.nan)
        x_new = update(y_obs, x, mask=mask)
        assert x_new.isfinite().all()
        return torch.allclose(x_new, x, rtol=rtol, atol=atol)
