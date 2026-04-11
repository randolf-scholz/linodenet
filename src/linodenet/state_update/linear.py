r"""Linear filters."""

__all__ = [
    "LinearRNNCell",
    "LinearCell",
    "KalmanCell",
]

from math import sqrt
from typing import Optional

import torch
from torch import Tensor, nn
from torch.linalg import solve, solve_triangular
from torch.nn import functional as F

from linodenet.mappings import bijections, surjections
from linodenet.nn.containers import Constant
from linodenet.nn.parametrize import register_parametrization
from linodenet.nn.rezero import resolve_gate
from signatures import signature

from .base import StateUpdaterBase


class LinearRNNCell(StateUpdaterBase):
    r"""Linear state update.

    .. math:: F(y，x) =  Ux + Vy + b

    where $U$ and $V$ are learnable matrices, and $b$ is a learnable bias vector.
    """

    # PARAMETERS
    U: Tensor
    r"""PARAM: the hidden state matrix."""
    V: Tensor
    r"""PARAM: the observable matrix."""
    bias: Optional[Tensor]
    r"""PARAM: the bias vector."""

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__(input_size, hidden_size)
        m = self.hidden_size
        n = self.input_size
        self.U = nn.Parameter(torch.normal(0, 1 / sqrt(m), size=(m, m)))
        self.V = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(m, n)))
        self.bias = nn.Parameter(torch.zeros(m)) if bool(bias) else None

    @signature("[(..., n), (..., m)] -> (..., m)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Forward pass of the state update.

        .. math:: F(y，x) =  Ux + Vy + b
        """
        return F.linear(x, self.U, None) + F.linear(y, self.V, self.bias)


class LinearCell(StateUpdaterBase):
    r"""Linear innovation state update.

    .. math:: x' = x - ρ(K(x)(y - h(x)))

    where $K(x)$ is a learnable innovation gain, $h$ is the observation map, and
    $ρ$ is a gate applied to the innovation correction. By default, $K$ is a
    learned constant matrix, but it can also be provided as a custom module that
    depends on the current hidden state $x$.

    The gain can be:

    - ``"constant"``: use a learned constant gain matrix. This is the default.
    - ``nn.Module``: use a custom user-provided state-dependent gain module.

    Standard gate options are:

    - ``"rezero"``: use a learnable ReZero scalar $ρ(z)=αz$ with $α$ initialized
      to zero, so that the cell starts as the identity map.
    - ``"identity"``: use $ρ(z)=z$ with no additional scaling.
    - ``None``: alias for ``"identity"``.
    - ``nn.Module``: use a custom user-provided gate.

    The observation map can be:

    - ``"linear"``: use a learned linear observation map.
    - ``"identity"``: use $h(x)=x$, which requires ``input_size == hidden_size``.
    - ``nn.Module``: use a custom user-provided observation map.
    """

    # PARAMETERS
    gain: nn.Module
    r"""MODULE: The innovation gain producing matrices $K(x)$."""
    observation_map: nn.Module
    r"""MODULE: The observation map used in the innovation term."""
    gate: nn.Module
    r"""MODULE: Optional gate for the innovation term."""

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        gain: str | nn.Module = "constant",
        gate: str | nn.Module | None = "rezero",
        observation_map: str | nn.Module = "linear",
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        self.gate = resolve_gate(gate)

        match gain:
            case nn.Module():
                self.gain = gain
            case "constant":
                self.gain = Constant((hidden_size, input_size))
            case str():
                raise ValueError(
                    f"Unknown gain: {gain!r}. Expected 'constant' or an nn.Module."
                )
            case _:
                raise TypeError(
                    f"gain must be a string or nn.Module, got {type(gain)!r}."
                )

        match observation_map:
            case nn.Module():
                self.observation_map = observation_map
            case "linear":
                self.observation_map = nn.Linear(hidden_size, input_size, bias=False)
            case "identity":
                if input_size != hidden_size:
                    raise ValueError(
                        "observation_map='identity' requires input_size == hidden_size!"
                    )
                self.observation_map = nn.Identity()
            case str():
                raise ValueError(
                    f"Unknown observation_map: {observation_map!r}. "
                    "Expected 'linear', 'identity', or an nn.Module."
                )
            case _:
                raise TypeError(
                    "observation_map must be a string or nn.Module, "
                    f"got {type(observation_map)!r}."
                )

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        y_pred = self.observation_map(x)
        r = torch.where(y.isnan(), 0.0, y - y_pred)  # (..., input_size)
        K = self.gain(x)  # (hidden_size, input_size) or (..., hidden_size, input_size)
        correction = (r.unsqueeze(-2) @ K.mT).squeeze(-2)
        return x - self.gate(correction)


class KalmanCell(StateUpdaterBase):
    r"""Linear Kalman-style hidden-state update with masked observations.

    .. math:: x' = x + ρ\left(Σ(x)HᵀMᵀ(M(HΣ(x)Hᵀ + R)Mᵀ)⁻¹(y - MHx)\right)

    Here, $y = Hx$, $Σ(x)$ is the hidden-state covariance, and
    $Σᵧᵧ = HΣ(x)Hᵀ + R$, for the masked observation model $y_{\text{obs}} = My$.
    In the implementation, $Σ(x)$ is represented through a covariance factor
    $L(x)$, typically a Cholesky factor, such that $Σ(x)=L(x)L(x)ᵀ$.
    $ρ$ is an optional gate applied to the Kalman correction. Standard
    gate options are the same as for `LinearInnovationCell`: ``"rezero"``,
    ``"identity"``, ``None``, or a custom `nn.Module`.

    Notes:
        LMMSE stands for linear minimum mean squared error: the best affine
        estimator under squared loss among estimators linear in the observations.
        BLUP stands for best linear unbiased predictor: the minimum-variance
        unbiased estimator within the same linear class.
    """

    observation_map: nn.Module
    r"""MODULE: Observation map $H$ from hidden to observation space."""
    covariance_factor: nn.Module
    r"""MODULE: Covariance factor $L(x)$ with $Σₓₓ(x)=L(x)L(x)ᵀ$."""
    noise_cholesky: Tensor
    r"""PARAM: Cholesky factor defining the observation noise covariance $R$."""
    gate: nn.Module
    r"""MODULE: Optional gate for the Kalman correction."""
    eye: Tensor
    r"""BUFFER: Identity matrix used to keep the covariance solve well-posed."""

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "noise": self.noise,
            "gate": self.gate,
        }

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        noise: str = "scalar",
        covariance_factor: str | nn.Module = "constant",
        gate: str | nn.Module | None = "rezero",
        observation_map: str | nn.Module = "linear",
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        m = self.hidden_size
        n = self.input_size
        self.gate = resolve_gate(gate)
        self.register_buffer("eye", torch.eye(n), persistent=False)

        match covariance_factor:
            case nn.Module():
                self.covariance_factor = covariance_factor
            case "constant":
                self.covariance_factor = Constant((m, m))
                register_parametrization(
                    self.covariance_factor,
                    "value",
                    surjections.CholeskyFactor(),
                )
            case str():
                raise ValueError(
                    "Unknown covariance_factor: "
                    f"{covariance_factor!r}. Expected 'constant' or an nn.Module."
                )
            case _:
                raise TypeError(
                    "covariance_factor must be a string or nn.Module, "
                    f"got {type(covariance_factor)!r}."
                )

        match observation_map:
            case nn.Module():
                self.observation_map = observation_map
            case "linear":
                self.observation_map = nn.Linear(hidden_size, input_size, bias=False)
            case "identity":
                if input_size != hidden_size:
                    raise ValueError(
                        "observation_map='identity' requires input_size == hidden_size!"
                    )
                self.observation_map = nn.Identity()
            case str():
                raise ValueError(
                    f"Unknown observation_map: {observation_map!r}. "
                    "Expected 'linear', 'identity', or an nn.Module."
                )
            case _:
                raise TypeError(
                    "observation_map must be a string or nn.Module, "
                    f"got {type(observation_map)!r}."
                )

        match noise:
            case "scalar":
                self.noise_cholesky = nn.Parameter(torch.zeros(()))
                register_parametrization(
                    self,
                    "noise_cholesky",
                    bijections.PositiveScalarMatrix(size=n),
                    unsafe=True,
                )
            case "diagonal":
                self.noise_cholesky = nn.Parameter(torch.normal(0, 1, size=(n,)))
                register_parametrization(
                    self,
                    "noise_cholesky",
                    bijections.PositiveDiagonal(),
                    unsafe=True,
                )
            case str():
                raise ValueError(
                    f"Unknown noise: {noise!r}. Expected 'scalar' or 'diagonal'."
                )
            case _:
                raise TypeError(f"noise must be a string, got {type(noise)!r}.")

    @signature("[(..., n), (..., m)] -> (..., m)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        y_pred = self.observation_map(x)
        missing = y.isnan()
        observed = (~missing).to(y.dtype)
        # TODO: consider solving only over unmasked coordinates (requires flattening).
        L = self.covariance_factor(x)
        # mask columns for unobserved values in cholesky factor
        J = torch.where(missing.unsqueeze(-2), self.eye, self.noise_cholesky)

        # Restrict the innovation y_obs - MHμₓ to the observed coordinates.
        innovation = torch.where(missing, torch.zeros_like(y_pred), y - y_pred)

        # Build HL indirectly as (LHᵀ)ᵀ to avoid depending on raw layer weights.
        HL = self.observation_map(L.mT).mT

        # u = (M(HLLᵀHᵀ + JJᵀ)M + I_missing)⁻¹r
        # note: M(HLLᵀHᵀ + JJᵀ)M + I_missing = J(𝕀 + BBᵀ)Jᵀ, B = J⁻¹MHL
        # solve via: z = J⁻¹r, w = (𝕀 + BBᵀ)⁻¹z, u = J⁻ᵀw
        # middle part via woodbury: (𝕀 + BBᵀ)⁻¹ = 𝕀 - B(𝕀 + BᵀB)⁻¹Bᵀ (good if m>n)
        B = solve_triangular(J, observed.unsqueeze(-1) * HL, upper=False)  # J⁻¹MHL
        z = solve_triangular(J, innovation.unsqueeze(-1), upper=False)  # J⁻¹r
        w = solve(self.eye + B @ B.mT, z)
        u = solve_triangular(J.mT, w, upper=True).squeeze(-1)  # J⁻ᵀw

        # correction = Σₓᵧu = LLᵀHᵀu
        correction = (u @ HL) @ L.mT

        return x + self.gate(correction)
