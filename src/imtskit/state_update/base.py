r"""Protocols and wrappers for state-update models.

A state update is a map $x' = F(y, x)$, with observation $y$ and current state
estimate $x$. The square case `input_size == hidden_size` is common but
not required.

Covariates $u$, such as control inputs or forecasts from another system, supply
additional conditioning information. They may be passed separately by a richer
interface or included in an enlarged state; the protocols here take two positional
inputs.

For example, a linear state-space model without noise satisfies

$$\begin{aligned}
    ẋ(t) &= A(t)x(t) + B(t)u(t), \\
    y(t) &= C(t)x(t) + D(t)u(t).
\end{aligned}$$

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
    "is_consistent_update",
]

from collections.abc import Callable, Iterable
from typing import Final, Protocol, TypeIs, cast

import torch
from torch import Tensor, nn

from imtskit.nn import ModuleSequence
from imtskit.nn.rezero import resolve_gate
from imtskit.testing.utils import get_device
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

    $$ x' = F(y, x) $$
    """

    def __call__(self, y: Y, x: X, /) -> X: ...


class VectorStateUpdate(AbstractStateUpdate[Tensor, Tensor], Protocol):
    r"""Protocol for vector-valued state updaters.

    $$ x' = F(y, x) $$
    """

    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(self, /, input_size: int, hidden_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size

    @signature("[(..., d), (..., h)] -> (..., h)")
    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor: ...


class SparseVectorStateUpdate(VectorStateUpdate, Protocol):
    r"""Protocol for vector-valued state updaters with observation masks.

    A true mask entry marks an observed coordinate. Concrete implementations
    determine how to infer a mask when none is supplied.
    """

    @signature("[(..., d), (..., h), (..., d)?] -> (..., h)")
    def __call__(
        self, y: Tensor, x: Tensor, /, *, mask: Tensor | None = None
    ) -> Tensor:
        r"""Update the state vector using an optional observation mask.

        Args:
            y: Observation vector.
            x: Current state estimate.
            mask: Boolean mask with true entries at observed coordinates. If `None`,
                mask inference is implementation-specific; some updaters require
                complete observations.
        """
        ...


def is_vector_state_updater(arg: object, /) -> TypeIs[VectorStateUpdate]:
    r"""Check for a callable with integer `input_size` and `hidden_size` attributes.

    This structural check does not validate the call signature or output shape.
    """
    return (
        callable(arg)
        and isinstance(getattr(arg, "input_size", None), int)
        and isinstance(getattr(arg, "hidden_size", None), int)
    )


# TODO: use Intersection type
class CellSequence[C: VectorStateUpdate](ModuleSequence[C], VectorStateUpdate):  # type: ignore
    r"""Apply multiple state updaters sequentially to the same observation.

    $$ xₖ₊₁ = Fₖ(y, xₖ) $$

    The sequence must be nonempty, and all cells must have the same input and
    hidden-state sizes.
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
    r"""Apply sequential state updates with subtractive residual corrections.

    $$ xₖ₊₁ = xₖ - αₖFₖ(y, xₖ) $$

    With `use_rezero=True`, the scales $αₖ$ are learnable and initialized to
    zero, so the sequence initially leaves the state unchanged. Otherwise, the
    scales are fixed at one and each stage subtracts its cell output.

    Args:
        cells: State-updater modules with matching input and hidden-state sizes.
        use_rezero: Whether to use learnable residual scales initialized to zero.
    """

    alpha: Tensor
    r"""PARAM: Residual scaling factors $αₖ$."""
    use_rezero: Tensor
    r"""Legacy flag annotation; only residual scales are stored by the constructor."""

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

    $$ x' = x - ρ(F(y, x)) $$

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
    r"""Impute missing observations before passing them to a state updater.

    $$ u = \op{impute}(y, x; m), \qquad x' = F(u, x) $$

    This wrapper uses true mask entries for values to impute. This is the opposite
    of the observation-mask convention in `SparseVectorStateUpdate`.
    If no mask is supplied, missing entries are identified by `y.isnan()`.

    Supported values of `imputation` are:

    - `"zero"`: replace missing entries with zero.
    - `"last"`: use the previous value, initially zero.
    - `"learnable"`: use a learnable vector.
    - `"linear"`: predict replacement values from the hidden state with a linear map.
    - A float or tensor: use a fixed replacement value.
    - An `nn.Module`: use a custom imputer accepting `(y, x, mask=mask)`.

    When `concat_mask=True` (the default), the updater receives the imputed vector
    followed by the missing-value mask:

    $$ u' = \op{concat}(u, m) $$

    The wrapped updater must then have twice the observation input size. Without
    mask concatenation, its input size must match the observation size.
    """

    state_updater: F
    imputer: ImputerProtocol

    # CONSTANTS
    concat_mask: Final[bool]
    r"""CONST: Whether to append the missing-value mask to the imputed input."""
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
        self.concat_mask = concat_mask

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


def is_consistent_update(
    update: SparseVectorStateUpdate,
    /,
    *,
    decoder: Callable[[Tensor], Tensor] | None = None,
    batch_shape: tuple[int, ...] = (8,),
    check_sparse_observations: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    r"""Check the decoder-consistency condition on random data.

    This checks the fixed-point criterion $ϕ(x)=y ⟹ F(y, x)=x$, where $ϕ$ is an
    optional decoder from hidden state to observation space. If no decoder is
    provided, the identity map is used, so `update` must be square.

    Args:
        update: The state updater to test.
        decoder: Optional decoder $ϕ$ mapping hidden states to observations. If
            `None`, use the identity map.
        batch_shape: Optional leading batch dimensions for the random test data.
        check_sparse_observations: Whether to also test the sparse-observation
            case where only a random subset of coordinates is observed.
            The updater must interpret true mask entries as observed coordinates.
        rtol: Relative tolerance for the equality check.
        atol: Absolute tolerance for the equality check.

    Returns:
        `True` if the update satisfies the criterion on the sampled data, else
        `False`.

    Raises:
        ValueError: If `decoder` is `None` and the updater is not square, or if
            `decoder` returns a tensor with the wrong shape.
        AssertionError: The sparse-observation update produces nonfinite values.
    """
    if decoder is None and update.input_size != update.hidden_size:
        raise ValueError(
            "Identity decoding requires a square state updater with "
            f"{update.input_size=} and {update.hidden_size=}."
        )

    device = get_device(update)
    x = torch.randn(*batch_shape, update.hidden_size, device=device)
    with torch.no_grad():
        y = x.clone() if decoder is None else decoder(x)
        if y.shape != (*batch_shape, update.input_size):
            raise ValueError(
                "decoder must map hidden states to observation tensors with shape "
                f"{(*batch_shape, update.input_size)}, but got {tuple(y.shape)}."
            )

        # check that ϕ(x)=y ⟹ F(y, x)=x
        if not torch.allclose(update(y, x), x, rtol=rtol, atol=atol):
            return False

        if not check_sparse_observations:
            return True

        # check that ϕ(x)=y ⟹ F([m ? y : NaN], x)=x for a random binary mask m
        mask = torch.rand(*batch_shape, update.input_size, device=device) > 0.5
        y_obs = torch.where(mask, y, torch.nan)
        x_new = update(y_obs, x, mask=mask)
        assert x_new.isfinite().all()
        return torch.allclose(x_new, x, rtol=rtol, atol=atol)
