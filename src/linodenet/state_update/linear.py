r"""Linear filters."""

__all__ = [
    "LinearCell",
    "LinearInnovationCell",
    "LinearKalmanCell",
]

from math import sqrt
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from linodenet.nn.rezero import resolve_gate
from signatures import signature

from .base import StateUpdaterBase


class LinearCell(StateUpdaterBase):
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


class LinearInnovationCell(StateUpdaterBase):
    r"""Linear innovation state update.

    .. math:: x' = x - ρ(K(y - h(x)))

    where $K$ is a learnable innovation gain, $h$ is the observation map, and
    $ρ$ is a gate applied to the innovation correction.

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
    gain: nn.Linear
    r"""MODULE: The learnable innovation gain."""
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
        gate: str | nn.Module | None = "rezero",
        observation_map: str | nn.Module = "linear",
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)

        self.gain = nn.Linear(input_size, hidden_size, bias=False)
        self.gate = resolve_gate(gate)

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
        r = y - self.observation_map(x)
        return x - self.gate(self.gain(r))


class LinearKalmanCell(StateUpdaterBase):
    r"""Linear Kalman-style hidden-state update with masked observations.

    Let $μ_x = x$, $μ_y = Hμ_x$, $Σₓₓ = \Cov(x)$, and
    $Σᵧᵧ = HΣₓₓHᵀ + R$. For the masked observation model

    .. math:: y_{\text{obs}} = My

    this cell computes the LMMSE / BLUP estimate

    .. math:: μₓ' = μₓ + Σₓₓ Hᵀ Mᵀ (M(HΣₓₓHᵀ + R)Mᵀ)⁻¹(y_{\text{obs}} - MHμₓ).
    """

    observation_map: nn.Linear
    r"""MODULE: Linear observation map $H$ from hidden to observation space."""
    state_scale: Tensor
    r"""PARAM: Factor defining the hidden covariance $Σₓₓ$."""
    noise_scale: Tensor
    r"""PARAM: Factor defining the observation noise covariance $R$."""
    eye: Tensor
    r"""BUFFER: Identity matrix used to keep the covariance solve well-posed."""

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        m = self.hidden_size
        n = self.input_size
        self.observation_map = nn.Linear(hidden_size, input_size, bias=False)
        self.state_scale = nn.Parameter(torch.normal(0, 1 / sqrt(m), size=(m, m)))
        self.noise_scale = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(n, n)))
        self.register_buffer("eye", torch.eye(n), persistent=False)

    @signature("[(..., n), (..., m)] -> (..., m)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        y_pred = self.observation_map(x)
        missing = y.isnan()
        observed = (~missing).to(y.dtype)

        # Build Σₓₓ, Σₓᵧ = ΣₓₓHᵀ, and Σᵧᵧ = HΣₓₓHᵀ.
        sigma_xx = self.state_scale @ self.state_scale.mT
        H = self.observation_map.weight
        sigma_xy = sigma_xx @ H.mT
        sigma_yy = H @ sigma_xx @ H.mT
        noise = self.noise_scale @ self.noise_scale.mT
        system = sigma_yy + noise + torch.finfo(y.dtype).eps * self.eye

        # Restrict the innovation y_obs - MHμₓ to the observed coordinates.
        innovation = torch.where(missing, torch.zeros_like(y_pred), y - y_pred)
        # Mask the normal equations to M(Σᵧᵧ + R)Mᵀ and keep the solve nonsingular
        # by inserting 𝕀 on unobserved coordinates.
        system_mask = observed.unsqueeze(-1) * observed.unsqueeze(-2)
        system = system * system_mask + torch.diag_embed(missing.to(y.dtype))
        # Apply the same projection to Σₓᵧ = ΣₓₓHᵀ.
        sigma_xy = sigma_xy * observed.unsqueeze(-2)

        gain_rhs = torch.linalg.solve(system, innovation.unsqueeze(-1))
        correction = (sigma_xy @ gain_rhs).squeeze(-1)

        return x + correction
