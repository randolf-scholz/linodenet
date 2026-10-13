r"""Linear filters."""

__all__ = [
    "LinearRNNCell",
    "ConstantGain",
    "AttentionGain",
    "AttentionCovarianceFactor",
    "InnovationCell",
    "KalmanCell",
]

from math import sqrt
from typing import Optional

import torch
from torch import Tensor, nn
from torch.linalg import solve, solve_triangular
from torch.nn import functional as F

from imtskit.mappings import bijections, surjections
from imtskit.nn.containers import Constant
from imtskit.nn.parametrize import register_parametrization
from imtskit.nn.rezero import ReZero, resolve_gate
from signatures import signature

from .base import SparseVectorStateUpdate, VectorStateUpdate


class LinearRNNCell(nn.Module, VectorStateUpdate):
    r"""Compute an affine linear state update.

    $$ F(y, x) = Ux + Vy + b $$

    Here $U$ and $V$ are learnable state and observation matrices. The learnable
    bias vector $b$ is omitted when `bias=False`.
    """

    # PARAMETERS
    U: Tensor
    r"""PARAM: Hidden-state matrix $U$."""
    V: Tensor
    r"""PARAM: Observation-to-state matrix $V$."""
    bias: Optional[Tensor]
    r"""PARAM: Optional affine bias vector."""

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
        self.bias = nn.Parameter(torch.zeros(m)) if bias else None

    @signature("[(..., n), (..., m)] -> (..., m)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Forward pass of the state update.

        $$ F(y,x) =  Ux + Vy + b $$
        """
        return F.linear(x, self.U, None) + F.linear(y, self.V, self.bias)


class InnovationCell(nn.Module, SparseVectorStateUpdate):
    r"""State update that is linear/affine in the residual $h(x)-y$.

    $$ x' = x - ρ(K(x)⋅(h(x) - y)) $$

    where $K(x)$ is a learnable innovation gain, $h$ is the observation map, and
    $ρ$ is a gate applied to the innovation correction. By default, $K$ is a
    learned constant matrix, but it can also be provided as a custom module that
    depends on the current hidden state $x$.

    The gain can be:

    - `"constant"`: use a learned constant gain matrix. This is the default.
    - `"attention"`: predict the gain matrix from $x$ with attention.
    - `nn.Module`: use a custom user-provided state-dependent gain module.

    Standard gate options are:

    - `"rezero"`: use a learnable ReZero scalar $ρ(z)=αz$ with $α$ initialized
        to zero, so that the cell starts as the identity map.
    - `"identity"`: use $ρ(z)=z$ with no additional scaling.
    - `None`: alias for `"identity"`.
    - `nn.Module`: use a custom user-provided gate.

    A custom gain module is called as `gain(residual, x)` and must return a
    correction with the hidden-state shape. A supplied mask marks observed
    coordinates and zeroes unobserved residuals. Without a mask, NaNs are not
    inferred as missing values.

    The observation map must be an `nn.Module`. If it is omitted, this cell uses
    $h(x)=x$, which requires `input_size == hidden_size`.
    """

    # PARAMETERS
    gain: nn.Module
    r"""MODULE: The innovation gain producing $K(x)r$."""
    observation_map: nn.Module
    r"""MODULE: The observation map used in the innovation term."""
    gate: nn.Module
    r"""MODULE: Optional gate for the innovation term."""

    @classmethod
    def from_direct_observation_model(
        cls, /, size: int, *, gate: str | nn.Module | None = "last-value"
    ) -> InnovationCell:
        r"""Construct a square direct-observation linear cell.

        This constructor fixes $h(x)=x$ and initializes $K(x)=I$, yielding

        $$ x' = x - ρ(x - y) $$

        Special gate presets initialize a learnable ReZero scalar to recover
        standard observation blends at construction time:

        - `"last-value"`: initialize with $α=1$, so $x' = y$.
        - `"average-value"`: initialize with $α=1/2$, so $x' = (x+y)/2$.
        - `"first-value"` / `"keep-state"`: initialize with $α=0$, so
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

        cell = InnovationCell(
            size,
            size,
            gain="constant",
            gate=gate_module,
            observation_map=nn.Identity(),
        )
        with torch.no_grad():
            assert isinstance(cell.gain, ConstantGain)
            cell.gain.weight.copy_(torch.eye(size))
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
        observation_map: nn.Module | None = None,
    ) -> None:
        super().__init__()
        SparseVectorStateUpdate.__init__(
            self, input_size=input_size, hidden_size=hidden_size
        )
        self.gate = resolve_gate(gate)

        match gain:
            case nn.Module():
                self.gain = gain
            case "constant":
                self.gain = ConstantGain(input_size, hidden_size)
            case "attention":
                self.gain = AttentionGain(
                    input_size,
                    hidden_size,
                    context_size=hidden_size,
                )
            case _:
                raise ValueError(
                    "Unknown gain: "
                    f"{gain!r}. Expected 'constant', 'attention', or an nn.Module."
                )

        match observation_map:
            case nn.Module():
                self.observation_map = observation_map
            case None if input_size == hidden_size:
                self.observation_map = nn.Identity()
            case None:
                raise ValueError(
                    "observation_map is required unless input_size == hidden_size."
                )
            case _:
                raise TypeError(
                    "observation_map must be an nn.Module; omit it only when "
                    "input_size == hidden_size to use nn.Identity()."
                )

    def forward(self, y: Tensor, x: Tensor, /, mask: Tensor | None = None) -> Tensor:
        assert y.shape[-1] == self.input_size
        assert x.shape[-1] == self.hidden_size
        r = self.observation_map(x) - y

        if mask is not None:
            r = torch.where(mask, r, 0.0)  # (..., input_size)

        correction = self.gain(r, x)
        return x - self.gate(correction)


class ConstantGain(nn.Module):
    r"""Apply a learned constant gain matrix to a residual vector.

    The map is $(v, x) ↦ Kv$; the context argument $x$ is unused.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, v: Tensor, _: Tensor, /) -> Tensor:
        return F.linear(v, self.weight)


class AttentionGain(nn.Module):
    r"""Predict a gain matrix with scaled dot-product attention.

    For context vector $x$, the gain entries are

    $$ Kᵢⱼ(x) = \softmaxⱼ\left(\frac{qᵢ(x)ᵀkⱼ(x)}{\sqrt{dₐ}}\right) $$

    Here $qᵢ(x)$ and $kⱼ(x)$ are row- and column-specific query and key vectors,
    and $dₐ$ is their feature size. Softmax is taken over input columns, so each
    gain row sums to one. The module returns $K(x)v$ for residual vector $v$.
    """

    query_proj: nn.Linear
    r"""MODULE: Projects the context vector to flattened query heads."""
    key_proj: nn.Linear
    r"""MODULE: Projects the context vector to flattened key heads."""
    input_size: int
    r"""CONST: Number of columns in the gain matrix."""
    output_size: int
    r"""CONST: Number of rows in the gain matrix."""
    context_size: int
    r"""CONST: Size of the context vector conditioning the gain."""
    num_heads: int
    r"""CONST: Number of attention heads used to score gain entries."""
    head_dim: int
    r"""CONST: Per-head query/key feature dimension."""
    hidden_size: int
    r"""CONST: Backward-compatible alias for the per-head query/key dimension."""

    @property
    def query(self) -> nn.Linear:
        return self.query_proj

    @property
    def key(self) -> nn.Linear:
        return self.key_proj

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "context_size": self.context_size,
            "hidden_size": self.hidden_size,
        }

    def __init__(
        self,
        /,
        input_size: int,
        output_size: int,
        *,
        context_size: int | None = None,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = self.output_size if context_size is None else context_size
        self.num_heads = 1
        self.hidden_size = (
            min(self.input_size, self.output_size, self.context_size, 32)
            if hidden_size is None
            else hidden_size
        )
        self.head_dim = self.hidden_size
        if self.context_size <= 0:
            raise ValueError("context_size must be a positive integer.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer.")

        query_features = self.output_size * self.num_heads * self.head_dim
        key_features = self.input_size * self.num_heads * self.head_dim

        # The context emits one query vector per output row and one key vector
        # per input column, flattened across heads.
        self.query_proj = nn.Linear(
            self.context_size,
            query_features,
            bias=False,
        )
        self.key_proj = nn.Linear(
            self.context_size,
            key_features,
            bias=False,
        )

    def forward(self, v: Tensor, x: Tensor, /) -> Tensor:
        # (..., output_size, H, d_h)
        q = (
            self.query_proj(x)  # (..., output_size * H * d_h)
            .unflatten(-1, (self.output_size, self.num_heads, self.head_dim))
            .swapaxes(-2, -3)  # (..., H, output_size, d_h)
        )
        k = (
            self.key_proj(x)  # (..., input_size * H * d_h)
            .unflatten(-1, (self.input_size, self.num_heads, self.head_dim))
            .swapaxes(-2, -3)  # (..., H, input_size, d_h)
        )
        v = v[..., None, None].swapaxes(-2, -3)  # (..., H, input_size, 1)

        # (..., H, output_size, 1)  (note: H=1)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return attended.squeeze(-3).squeeze(-1)  # (..., output_size)


class KalmanCell(nn.Module, SparseVectorStateUpdate):
    r"""Update a hidden state with a gated, masked Kalman correction.

    Let $H = 𝐃h(x)$ be the Jacobian of the observation map at the current state,
    $Σ = L(x)L(x)ᵀ$ the learned state covariance, and $R$ the observation noise
    covariance. For a selector $Πₘ$ onto observed coordinates, the update is

    $$\begin{aligned}
        Kₘ &= ΣHᵀΠₘᵀ\left[Πₘ(HΣHᵀ + R)Πₘᵀ\right]⁻¹ \\
        x' &= x - ρ\left(KₘΠₘ(h(x) - y)\right)
    \end{aligned}$$

    The module obtains $HL(x)$ by applying Jacobian-vector products to the
    columns of the covariance factor. It performs the masked solve in the full
    observation space, zeroing unobserved residuals and Jacobian rows and replacing
    unobserved noise blocks with the identity.

    A supplied mask uses true entries for observed coordinates. Missing values
    must be masked explicitly; without a mask, NaNs propagate into the solve.
    The gate options match `InnovationCell`: `"rezero"`, `"identity"`,
    `None`, or a custom `nn.Module`.

    For nonlinear $h$, the Jacobian gives the local linearization used by an
    extended Kalman filter. This cell updates the state vector; it does not
    return an updated covariance.

    Notes:
        LMMSE means linear minimum mean squared error, the best affine estimator
        under squared loss. BLUP means best linear unbiased predictor, the
        minimum-variance unbiased estimator within the linear class.
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
        observation_map: nn.Module | None = None,
    ) -> None:
        super().__init__()
        SparseVectorStateUpdate.__init__(
            self, input_size=input_size, hidden_size=hidden_size
        )
        m = self.hidden_size
        n = self.input_size
        self.gate = resolve_gate(gate)
        self.register_buffer("eye", torch.eye(n), persistent=False)

        match covariance_factor:
            case nn.Module():
                self.covariance_factor = covariance_factor

            case "constant":
                value = torch.randn((m, m))
                self.covariance_factor = Constant(value, learnable=True)
                register_parametrization(
                    self.covariance_factor,
                    "value",
                    surjections.CholeskyFactor(),
                )

            case "attention":
                self.covariance_factor = AttentionCovarianceFactor(m)

            case _:
                raise ValueError(
                    "Unknown covariance_factor: "
                    f"{covariance_factor!r}. Expected 'constant', 'attention', or an nn.Module."
                )

        match observation_map:
            case nn.Module():
                self.observation_map = observation_map
            case None if input_size == hidden_size:
                self.observation_map = nn.Identity()
            case None:
                raise ValueError(
                    "observation_map is required unless input_size == hidden_size."
                )
            case _:
                raise TypeError(
                    "observation_map must be an nn.Module; omit it only when "
                    "input_size == hidden_size to use nn.Identity()."
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

            case _:
                raise ValueError(
                    f"Unknown noise: {noise!r}. Expected 'scalar' or 'diagonal'."
                )

    @signature("[(..., n), (..., m)] -> (..., m)")
    def forward(self, y: Tensor, x: Tensor, /, mask: Tensor | None = None) -> Tensor:
        assert y.shape[-1] == self.input_size
        assert x.shape[-1] == self.hidden_size
        *batch_shape, _ = x.shape

        y_pred, jvp_fn = torch.func.linearize(self.observation_map, x)
        batched_jvp_fn = torch.func.vmap(jvp_fn, -1, -1)

        # TODO: consider solving only over unmasked coordinates (requires flattening).
        L = self.covariance_factor(x).expand(
            *batch_shape, self.hidden_size, self.hidden_size
        )
        # Push the covariance-factor columns through 𝐃h(x) to obtain 𝐃h(x)L(x).
        HL = batched_jvp_fn(L)
        J = self.noise_cholesky.expand(*batch_shape, self.input_size, self.input_size)
        r = y_pred - y

        if mask is not None:
            # Restrict the residual r = Mh(x) - y_obs to the observed coordinates.
            r = torch.where(mask, r, 0.0)
            # Replace unobserved noise blocks by the identity to keep the solve well-posed.
            J = torch.where(mask.unsqueeze(-2), J, self.eye)
            HL = torch.where(mask[..., None], HL, 0.0)  # shape: (..., n, m)

        assert L.shape == (*batch_shape, self.hidden_size, self.hidden_size)
        assert J.shape == (*batch_shape, self.input_size, self.input_size)
        assert HL.shape == (*batch_shape, self.input_size, self.hidden_size)

        # u = (M(𝐃h(x)LLᵀ𝐃h(x)ᵀ + JJᵀ)M + I_missing)⁻¹r
        # note: M(𝐃h(x)LLᵀ𝐃h(x)ᵀ + JJᵀ)M + I_missing = J(𝕀 + BBᵀ)Jᵀ, B = J⁻¹M𝐃h(x)L
        # solve via: z = J⁻¹r, w = (𝕀 + BBᵀ)⁻¹z, u = J⁻ᵀw
        # middle part via woodbury: (𝕀 + BBᵀ)⁻¹ = 𝕀 - B(𝕀 + BᵀB)⁻¹Bᵀ (good if m>n)
        B = solve_triangular(J, HL, upper=False)  # J⁻¹M𝐃h(x)L (..., n, m)
        z = solve_triangular(J, r.unsqueeze(-1), upper=False)  # J⁻¹r
        w = solve(self.eye + B @ B.mT, z)  # shape: (..., n, 1)
        u = solve_triangular(J.mT, w, upper=True).squeeze(-1)  # J⁻ᵀw (..., n)
        assert u.shape == (*batch_shape, self.input_size)

        # δ = Σₓ₞u = L(x)L(x)ᵀ𝐃h(x)ᵀu
        d = torch.einsum("...n, ...nm, ...km -> ...k", u, HL, L)  # (..., m)

        return x - self.gate(d)


class AttentionCovarianceFactor(nn.Module):
    r"""Predict a Cholesky factor with attention-style pairwise interactions.

    Let $ϕᵢ(x) ∈ ℝ^{dₐ}$ denote the feature vector for row $i$. The module builds
    a lower-triangular factor $L(x)$ as

    $$
    Lᵢⱼ(x) =
    \begin{cases}
        \frac{ϕᵢ(x)ᵀϕⱼ(x)}{\sqrt{dₐ}} :& i > j \\
        \softplus(dᵢ(x)) + ε          :& i = j  \\
        0                             :& i < j,
    \end{cases}
    $$

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
            min(hidden_size, 32) if attention_size is None else attention_size
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
