r"""Linear filters."""

__all__ = [
    "LinearRNNCell",
    "AttentionGain",
    "AttentionCovarianceFactor",
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
from linodenet.nn.rezero import ReZero, resolve_gate
from signatures import signature

from .base import VectorStateUpdate


class LinearCell(nn.Module, VectorStateUpdate):
    r"""Linear innovation state update.

    .. math:: x' = x - ρ(K(x)⋅(h(x) - y))

    where $K(x)$ is a learnable innovation gain, $h$ is the observation map, and
    $ρ$ is a gate applied to the innovation correction. By default, $K$ is a
    learned constant matrix, but it can also be provided as a custom module that
    depends on the current hidden state $x$.

    The gain can be:

    - ``"constant"``: use a learned constant gain matrix. This is the default.
    - ``"attention"``: predict the gain matrix from $x$ with attention.
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

    @classmethod
    def from_direct_observation_model(
        cls, /, size: int, *, gate: str | nn.Module | None = "last-value"
    ) -> LinearCell:
        r"""Construct a square direct-observation linear cell.

        This constructor fixes $h(x)=x$ and initializes $K(x)=I$, yielding

        .. math:: x' = x - ρ(x - y).

        Special gate presets initialize a learnable ReZero scalar to recover
        standard observation blends at construction time:

        - ``"last-value"``: initialize with $α=1$, so $x' = y$.
        - ``"average-value"``: initialize with $α=1/2$, so $x' = (x+y)/2$.
        - ``"first-value"`` / ``"keep-state"``: initialize with $α=0$, so
          $x' = x$.
        """
        gate_module: str | nn.Module | None
        initial_gate_value: float | None = None

        match gate:
            case "last-value":
                gate_module = "rezero"
                initial_gate_value = 1.0
            case "average-value":
                gate_module = "rezero"
                initial_gate_value = 0.5
            case "first-value" | "keep-state":
                gate_module = "rezero"
                initial_gate_value = 0.0
            case None | "identity" | "rezero" | nn.Module():
                gate_module = gate
            case _:
                raise ValueError(
                    f"Unknown direct-observation gate: {gate!r}. Expected "
                    "'last-value', 'average-value', 'first-value', 'keep-state', "
                    "'rezero', 'identity', None, or an nn.Module."
                )

        cell = LinearCell(
            size,
            size,
            gain="constant",
            gate=gate_module,
            observation_map="identity",
        )
        with torch.no_grad():
            assert isinstance(cell.gain, Constant)
            cell.gain.value.copy_(torch.eye(size))
            if initial_gate_value is not None:
                assert isinstance(cell.gate, ReZero)
                cell.gate.scalar.copy_(torch.tensor(initial_gate_value))

        return cell

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
        super().__init__()
        VectorStateUpdate.__init__(self, input_size=input_size, hidden_size=hidden_size)
        self.gate = resolve_gate(gate)

        match gain:
            case nn.Module():
                self.gain = gain
            case "constant":
                value = torch.randn((hidden_size, input_size))
                self.gain = Constant(value, learnable=True)
            case "attention":
                self.gain = AttentionGain(hidden_size, input_size)
            case _:
                raise ValueError(
                    "Unknown gain: "
                    f"{gain!r}. Expected 'constant', 'attention', or an nn.Module."
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
            case _:
                raise ValueError(
                    f"Unknown observation_map: {observation_map!r}. "
                    "Expected 'linear', 'identity', or an nn.Module."
                )

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        y_pred = self.observation_map(x)
        r = torch.where(y.isnan(), 0.0, y_pred - y)  # (..., input_size)
        K = self.gain(x)  # (hidden_size, input_size) or (..., hidden_size, input_size)
        correction = (r.unsqueeze(-2) @ K.mT).squeeze(-2)
        return x - self.gate(correction)


class ConditionalLinear(nn.Module):
    r"""Computes $(v, x) ↦ K(x)v$."""

    def __init__(
        self,
        /,
        input_size: int,
        output_size: int,
        *,
        kind: str = "constant",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        match kind:
            case "constant":
                value = torch.randn((output_size, input_size))
                self.gain = Constant(value, learnable=True)
            case _:
                raise NotImplementedError

    def forward(self, v: Tensor, x: Tensor) -> Tensor:
        return F.linear(v, self.gain(x))


class AttentionGain(nn.Module):
    r"""Predict a gain matrix with scaled dot-product attention.

    For hidden state $x$, the gain entries are computed as

    .. math:: Kᵢⱼ(x) = \softmax_j(\frac{qᵢ(x)ᵀkⱼ(x)}{\sqrt{dₐ}})

    where $qᵢ(x)$ and $kⱼ(x)$ are row- and column-specific query/key vectors,
    and $dₐ$ is the shared attention feature size.
    """

    query: nn.Linear
    r"""MODULE: Projects the hidden state to row queries."""
    key: nn.Linear
    r"""MODULE: Projects the hidden state to column keys."""
    hidden_size: int
    r"""CONST: Number of rows in the gain matrix."""
    input_size: int
    r"""CONST: Number of columns in the gain matrix."""
    attention_size: int
    r"""CONST: Shared query/key feature dimension."""
    scale: float
    r"""CONST: Scale factor for attention logits."""

    @property
    def config(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "input_size": self.input_size,
            "attention_size": self.attention_size,
        }

    def __init__(
        self,
        /,
        hidden_size: int,
        input_size: int,
        *,
        attention_size: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_size = (
            min(hidden_size, input_size, 32)
            if attention_size is None
            else int(attention_size)
        )
        if self.attention_size <= 0:
            raise ValueError("attention_size must be a positive integer.")

        self.query = nn.Linear(
            hidden_size,
            hidden_size * self.attention_size,
            bias=False,
        )
        self.key = nn.Linear(
            hidden_size,
            input_size * self.attention_size,
            bias=False,
        )
        self.scale = self.attention_size**-0.5

    def forward(self, x: Tensor) -> Tensor:
        query = self.query(x).unflatten(-1, (self.hidden_size, self.attention_size))
        key = self.key(x).unflatten(-1, (self.input_size, self.attention_size))
        scores = self.scale * (query @ key.mT)
        return scores.softmax(dim=-1)


class KalmanCell(nn.Module, VectorStateUpdate):
    r"""Kalman-style hidden-state update with masked observations.

    .. math::
        x' = x - ρ\left(
            Σ(x)𝐃h(x)ᵀMᵀ (M(𝐃h(x)Σ(x)𝐃h(x)ᵀ + R)Mᵀ)⁻¹ (Mh(x) - y)
        \right)

    Here, $h(x)$ is the observation map, $𝐃h(x)$ is its local linearization at
    the current hidden state, and $Σ(x)$ is the hidden-state covariance. The
    masked observation model is $y_{\text{obs}} = My$, with local observation
    covariance $Σ₞₞(x) = 𝐃h(x)Σ(x)𝐃h(x)ᵀ + R$. In the implementation, $Σ(x)$ is
    represented through a covariance factor $L(x)$, typically a Cholesky
    factor, such that $Σ(x)=L(x)L(x)ᵀ$. The Jacobian action $𝐃h(x)L(x)$ is
    obtained by pushing the columns of $L(x)$ through the JVP of $h$ at $x$.
    $ρ$ is an optional gate applied to the Kalman correction. Standard gate
    options are the same as for `LinearInnovationCell`: ``"rezero"``,
    ``"identity"``, ``None``, or a custom `nn.Module`.

    Notes:
        LMMSE stands for linear minimum mean squared error: the best affine
        estimator under squared loss among estimators linear in the observations.
        BLUP stands for best linear unbiased predictor: the minimum-variance
        unbiased estimator within the same linear class.

    Remark:
        When $h$ is non-linear, this update uses the local Jacobian $𝐃h(x)$ at
        the current state, which is the same first-order linearization step used
        by the extended Kalman filter. In that sense, `KalmanCell` implements an
        EKF-style measurement update with a learned state covariance factor and
        optional gated correction.
    """

    observation_map: nn.Module
    r"""MODULE: Observation map $h$ from hidden to observation space."""
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
        super().__init__()
        VectorStateUpdate.__init__(self, input_size=input_size, hidden_size=hidden_size)
        m = self.hidden_size
        n = self.input_size
        self.gate = resolve_gate(gate)
        self.register_buffer("eye", torch.eye(n), persistent=False)

        match covariance_factor:
            case nn.Module():
                self.covariance_factor = covariance_factor

            case "constant":
                value = torch.randn((m, n))
                self.covariance_factor = Constant(value, learnable=True)
                register_parametrization(
                    self.covariance_factor,
                    "value",
                    surjections.CholeskyFactor(),
                )

            case "attention":
                self.covariance_factor = AttentionCovarianceFactor(m)

            case str():
                raise ValueError(
                    "Unknown covariance_factor: "
                    f"{covariance_factor!r}. Expected 'constant', 'attention', or an nn.Module."
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
        *batch_shape, _ = x.shape
        missing = y.isnan()

        y_pred, jvp_fn = torch.func.linearize(self.observation_map, x)

        # TODO: consider solving only over unmasked coordinates (requires flattening).
        L = self.covariance_factor(x).expand(
            *batch_shape, self.hidden_size, self.hidden_size
        )
        assert L.shape == (*batch_shape, self.hidden_size, self.hidden_size)

        # mask columns for unobserved values in cholesky factor
        J = torch.where(missing.unsqueeze(-2), self.eye, self.noise_cholesky)
        assert J.shape == (*batch_shape, self.input_size, self.input_size)

        # Restrict the residual r = Mh(x) - y_obs to the observed coordinates.
        r = torch.where(missing, torch.zeros_like(y_pred), y_pred - y)

        # Push the covariance-factor columns through 𝐃h(x) to obtain 𝐃h(x)L(x).
        batched_jvp_fn = torch.func.vmap(jvp_fn, -1, -1)
        MHL = ~missing[..., None] * batched_jvp_fn(L)  # shape: (..., n, m)
        assert MHL.shape == (*batch_shape, self.input_size, self.hidden_size)

        # u = (M(𝐃h(x)LLᵀ𝐃h(x)ᵀ + JJᵀ)M + I_missing)⁻¹r
        # note: M(𝐃h(x)LLᵀ𝐃h(x)ᵀ + JJᵀ)M + I_missing = J(𝕀 + BBᵀ)Jᵀ, B = J⁻¹M𝐃h(x)L
        # solve via: z = J⁻¹r, w = (𝕀 + BBᵀ)⁻¹z, u = J⁻ᵀw
        # middle part via woodbury: (𝕀 + BBᵀ)⁻¹ = 𝕀 - B(𝕀 + BᵀB)⁻¹Bᵀ (good if m>n)
        B = solve_triangular(J, MHL, upper=False)  # J⁻¹M𝐃h(x)L (..., n, m)
        z = solve_triangular(J, r.unsqueeze(-1), upper=False)  # J⁻¹r
        w = solve(self.eye + B @ B.mT, z)  # shape: (..., n, 1)
        u = solve_triangular(J.mT, w, upper=True).squeeze(-1)  # J⁻ᵀw (..., n)
        assert u.shape == (*batch_shape, self.input_size)

        # δ = Σₓ₞u = L(x)L(x)ᵀ𝐃h(x)ᵀu
        d = torch.einsum("...n, ...nm, ...km -> ...k", u, MHL, L)  # (..., m)

        return x - self.gate(d)


class LinearRNNCell(nn.Module, VectorStateUpdate):
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
        super().__init__()
        VectorStateUpdate.__init__(self, input_size=input_size, hidden_size=hidden_size)
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


class AttentionCovarianceFactor(nn.Module):
    r"""Predict a Cholesky factor with attention-style pairwise interactions.

    Let $ϕᵢ(x) ∈ ℝ^{dₐ}$ denote the feature vector for row $i$. The module builds
    a lower-triangular factor $L(x)$ as

    .. math:: Lᵢⱼ(x) =
        \begin{cases}
            \frac{ϕᵢ(x)^⊤ϕⱼ(x)}{\sqrt{dₐ}}, & i > j, \\
            \softplus(dᵢ(x)) + ε, & i = j, \\
            0, & i < j,
        \end{cases}

    where $dᵢ(x)$ are learned diagonal logits and $ε$ is machine epsilon for the
    current dtype, ensuring a strictly positive diagonal.
    """

    features: nn.Linear
    r"""MODULE: Projects the hidden state to shared attention features."""
    diagonal: nn.Linear
    r"""MODULE: Projects the hidden state to diagonal logits."""
    hidden_size: int
    r"""CONST: Number of rows and columns in the Cholesky factor."""
    attention_size: int
    r"""CONST: Shared attention feature dimension."""
    scale: float
    r"""CONST: Scale factor for the bilinear scores."""

    @property
    def config(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "attention_size": self.attention_size,
        }

    def __init__(
        self,
        /,
        hidden_size: int,
        *,
        attention_size: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_size = (
            min(hidden_size, 32) if attention_size is None else int(attention_size)
        )
        if self.attention_size <= 0:
            raise ValueError("attention_size must be a positive integer.")

        self.features = nn.Linear(
            hidden_size,
            hidden_size * self.attention_size,
            bias=False,
        )
        self.diagonal = nn.Linear(hidden_size, hidden_size)
        self.scale = self.attention_size**-0.5

        with torch.no_grad():
            self.diagonal.weight.zero_()
            self.diagonal.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x).unflatten(
            -1, (self.hidden_size, self.attention_size)
        )
        scores = self.scale * (features @ features.mT)
        offdiagonal = scores.tril(-1)
        diagonal = F.softplus(self.diagonal(x)) + torch.finfo(x.dtype).eps
        return offdiagonal + torch.diag_embed(diagonal)
