r"""State update modules."""

__all__ = [
    "AttentionCovarianceFactor",
    "AttentionGain",
    "ConstantGain",
    "InnovationCell",
    "KalmanCell",
    "LinearRNNCell",
    "GradientStepUpdater",
    "PositiveScalarMatrix",
    "LpLoss",
    "Constant",
    "lp_loss",
    "ReZero",
    "PositiveDiagonalMatrix",
    "CholeskyFactor",
]


from collections.abc import Callable
from functools import partial
from math import sqrt
from typing import Optional, cast

import torch
from torch import Tensor, nn
from torch.linalg import solve, solve_triangular
from torch.nn import functional as F


def lp_loss(
    x: Tensor,  # (..., d)
    y: Tensor,  # (..., d)
    /,
    *,
    mask: Tensor | None = None,  # (..., d)
    p: float = 2.0,
    dim: int = -1,
    aggregation: str = "sum",
) -> Tensor:  # (...)
    r"""Compute a per-batch-element $Lᵖ$ reconstruction loss $‖x-y‖ₚᵖ$."""
    r = x - y
    if mask is not None:
        r = torch.where(mask, r, 0.0)
        count = mask.sum(dim=dim)
    else:
        count = r.shape[-1]

    match aggregation:
        case "sum":
            return r.abs().pow(p).sum(dim=dim)
        case "mean":
            return r.abs().pow(p).sum(dim=dim).div(count)
        case _:
            raise ValueError(f"Unexpected aggregation: {aggregation!r}")


class LpLoss(nn.Module):
    r"""Compute a per-batch-element $Lᵖ$ reconstruction loss $‖x-y‖ₚᵖ$."""

    def __init__(
        self,
        p: float = 2.0,
        dim: int = -1,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        if p <= 0:
            raise ValueError(f"Expected p > 0, got {p!r}.")
        if aggregation not in {"sum", "mean"}:
            raise ValueError(
                f"Expected aggregation to be 'sum' or 'mean', got {aggregation!r}."
            )

        self.p = p
        self.dim = dim
        self.aggregation = aggregation

    __call__: Callable[[Tensor, Tensor], Tensor]

    def forward(self, x: Tensor, y: Tensor, /, *, mask: Tensor | None = None) -> Tensor:
        return lp_loss(
            x,
            y,
            mask=mask,
            p=self.p,
            dim=self.dim,
            aggregation=self.aggregation,
        )


class ReZero[
    M: nn.Module = nn.Module,
    S: nn.Module = nn.Module,
](nn.Module):
    r"""ReZero module.

    Simply multiplies the inputs by a scalar initialized to zero.
    """

    scalar: Tensor
    r"""PARAM: The scalar to multiply the inputs by."""
    scalar_map: S
    r"""MODULE: Map applied to the scalar before scaling the input."""
    module: M
    r"""MODULE: Map applied to the inputs before scaling them."""

    @property
    def config(self) -> dict:
        return {
            "module": self.module,
            "scalar": self.scalar,
            "scalar_map": self.scalar_map,
        }

    def __init__[U: nn.Module = nn.Identity, V: nn.Module = nn.Identity](
        self: ReZero[U, V],
        module: U | None = None,
        *,
        scalar_map: V | None = None,
        initial_value: Tensor | float = 0.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.scalar = nn.Parameter(
            torch.as_tensor(initial_value), requires_grad=learnable
        )
        self.module = cast("U", nn.Identity() if module is None else module)
        self.scalar_map = cast("V", nn.Identity() if scalar_map is None else scalar_map)

    # @signature("(..., *xs) -> (..., *xs)")
    def forward(self, x: Tensor) -> Tensor:
        return self.scalar_map(self.scalar) * self.module(x)

    # @signature("(..., *xs) -> (..., *xs)")
    def right_inverse(self, y: Tensor) -> Tensor | None:
        if getattr(self.module, "right_inverse", None) is None:
            return None

        return self.module.right_inverse(y / self.scalar_map(self.scalar))  # type: ignore[call]


class Constant(nn.Module):
    r"""Module that returns a learned constant tensor."""

    value: Tensor
    r"""PARAM: Constant tensor returned by the module."""

    def __init__(
        self,
        value: Tensor | float,
        /,
        *,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        match value:
            case float(value) | int(value):
                self.value = nn.Parameter(
                    torch.as_tensor(value), requires_grad=learnable
                )
            case Tensor() as tensor:
                self.value = nn.Parameter(tensor, requires_grad=learnable)
            case _:
                raise TypeError(f"Expected shape or tensor, got {type(value)!r}")

    def forward(self, _: Tensor) -> Tensor:
        return self.value


class GradientStepUpdater(nn.Module):
    r"""Single gradient-step updater for latent distribution parameters.

    .. math:: ℒ(z) = ∇₟ℓ(f(z), y) + λ d(z, z₋)
                z' = z₋ - η∇₟ℒ(z₋)
    """

    def __init__(
        self,
        *,
        decoder: nn.Module,
        loss: nn.Module | str = "l2",
        regularizer: nn.Module | str = "l2",
        regularization_strength: float = 1e-3,
        step_size: float = 1e-2,
    ) -> None:
        super().__init__()

        self.decoder = decoder
        self.regularization_strength = nn.Parameter(
            torch.as_tensor(regularization_strength)
        )
        self.step_size = nn.Parameter(torch.as_tensor(step_size))

        match loss:
            case nn.Module():
                self.loss = loss
            case "l1":
                self.loss = LpLoss(p=1.0)
            case "l2":
                self.loss = LpLoss(p=2.0)
            case _:
                raise ValueError(f"Unknown loss: {loss!r}")

        match regularizer:
            case nn.Module():
                self.regularizer = regularizer
            case "l1":
                self.regularizer = LpLoss(p=1.0)
            case "l2":
                self.regularizer = LpLoss(p=2.0)
            case _:
                raise ValueError(f"Unknown regularizer: {regularizer!r}")

    @partial(torch.func.vmap, in_dims=(None, 0, 0, 0))
    @partial(torch.func.grad, argnums=1)
    def _grad_fn_no_mask(
        self,
        z: Tensor,  # (B, d)
        z_prev: Tensor,  # (B, d)
        y: Tensor,  # (B, e)
        /,
    ) -> Tensor:  # (B)
        return (
            self.loss(self.decoder(z), y)  # ℓ(f(z), y)
            + self.regularization_strength * self.regularizer(z, z_prev)  # λ‖z-z₋‖²
        )

    @partial(torch.func.vmap, in_dims=(None, 0, 0, 0, 0))
    @partial(torch.func.grad, argnums=1)
    def _grad_fn_with_mask(
        self,
        z: Tensor,  # (B, d)
        z_prev: Tensor,  # (B, d)
        y: Tensor,  # (B, e)
        mask: Tensor,  # (B, e)
        /,
    ) -> Tensor:  # (B)
        return (
            # ℓ(f(z), y)
            self.loss(self.decoder(z), y, mask=mask)  # pyright: ignore[reportCallIssue]
            + self.regularization_strength * self.regularizer(z, z_prev)  # λ⋅‖z-z₋‖²
        )

    def grad_fn(
        self, z: Tensor, z_prev: Tensor, y: Tensor, /, *, mask: Tensor | None = None
    ) -> Tensor:
        r"""Return the gradient while preserving the input batch shape."""
        z_flat = z.reshape(-1, z.shape[-1])
        z_prev_flat = z_prev.reshape(-1, z_prev.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])

        grad = (
            self._grad_fn_no_mask(
                z_flat,
                z_prev_flat,
                y_flat,
            )
            if mask is None
            else self._grad_fn_with_mask(
                z_flat,
                z_prev_flat,
                y_flat,
                mask.reshape(-1, mask.shape[-1]),
            )
        )

        return grad.reshape_as(z)

    __call__: Callable[[Tensor, Tensor], Tensor]

    def forward(
        self,
        z: Tensor,  # (..., d)
        y: Tensor,  # (..., e)
        /,
        mask: Tensor | None = None,  # (..., e)
    ) -> Tensor:  # (..., d)
        r"""Computes z_prev - η∇₟ℒ(z_prev), where ℒ(z) = ℓ(f(z), y) + λ⋅d(z, z_prev)."""
        return z - self.step_size * self.grad_fn(z, z, y, mask=mask)


class LinearRNNCell(nn.Module):
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
        self.input_size = input_size
        self.hidden_size = hidden_size
        m = self.hidden_size
        n = self.input_size
        self.U = nn.Parameter(torch.normal(0, 1 / sqrt(m), size=(m, m)))
        self.V = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(m, n)))
        self.bias = nn.Parameter(torch.zeros(m)) if bool(bias) else None

    # @signature("[(..., n), (..., m)] -> (..., m)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Forward pass of the state update.

        .. math:: F(y，x) =  Ux + Vy + b
        """
        return F.linear(x, self.U, None) + F.linear(y, self.V, self.bias)


class InnovationCell(nn.Module):
    r"""State update that is linear/affine in the residual $h(x)-y$.

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
    r"""MODULE: The innovation gain producing $K(x)r$."""
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
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        match gate:
            case None | "identity":
                self.gate = nn.Identity()
            case "rezero":
                self.gate = ReZero()
            case nn.Module():
                self.gate = gate
            case _:
                raise ValueError(
                    f"Unknown gate: {gate!r}. Expected 'rezero', 'identity', or an nn.Module."
                )

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

    def forward(self, y: Tensor, x: Tensor, /, mask: Tensor | None = None) -> Tensor:
        assert y.shape[-1] == self.input_size
        assert x.shape[-1] == self.hidden_size
        r = self.observation_map(x) - y

        if mask is not None:
            r = torch.where(mask, r, 0.0)  # (..., input_size)

        correction = self.gain(r, x)
        return x - self.gate(correction)


class ConstantGain(nn.Module):
    r"""Computes $(v, x) ↦ K(x)v$."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.randn(input_size, output_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, v: Tensor, _: Tensor, /) -> Tensor:
        return F.linear(v, self.weight)


class AttentionGain(nn.Module):
    r"""Predict a gain matrix with scaled dot-product attention.

    For context vector $x$, the gain entries are computed as

    .. math:: Kᵢⱼ(x) = \softmax_j(\frac{qᵢ(x)ᵀkⱼ(x)}{\sqrt{dₐ}})

    where $qᵢ(x)$ and $kⱼ(x)$ are row- and column-specific query/key vectors,
    and $dₐ$ is the shared attention feature size.
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
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.context_size = (
            self.output_size if context_size is None else int(context_size)
        )
        self.num_heads = 1
        self.hidden_size = (
            min(self.input_size, self.output_size, self.context_size, 32)
            if hidden_size is None
            else int(hidden_size)
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


class PositiveScalarMatrix(nn.Module):
    r"""Parametrization of a positive scalar matrix $eᶜ𝕀$."""

    eye: Tensor

    def __init__(self, size: int, log_scale: Tensor | float = 0.0) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.as_tensor(log_scale))
        self.register_buffer("eye", torch.eye(size))

    def forward(self) -> Tensor:
        return self.eye * torch.exp(self.log_scale)


class PositiveDiagonalMatrix(nn.Module):
    r"""Parametrization of a positive diagonal matrix as $\diag(eᵛ)$."""

    def __init__(self, size: int, log_scales: Tensor | float = 0.0) -> None:
        super().__init__()
        self.register_buffer("eye", torch.eye(size))
        # store diagonal (v_1, v_2, ..., v_n)
        self.log_scales = nn.Parameter(torch.as_tensor(log_scales).expand(size))

    def forward(self) -> Tensor:
        return self.log_scales.diag_embed()


class CholeskyFactor(nn.Module):
    r"""Parametrize Cholesky factors via a lower-triangular matrix with log-diagonal."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size, size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, _, /) -> Tensor:
        return self.weight.tril(diagonal=-1) + torch.diag_embed(
            torch.exp(self.weight.diagonal(dim1=-2, dim2=-1))
        )


class KalmanCell(nn.Module):
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
    noise_cholesky: nn.Module
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
        self.input_size = input_size
        self.hidden_size = hidden_size
        m = self.hidden_size
        n = self.input_size
        self.register_buffer("eye", torch.eye(n), persistent=False)

        match gate:
            case None | "identity":
                self.gate = nn.Identity()
            case "rezero":
                self.gate = ReZero()
            case nn.Module():
                self.gate = gate
            case _:
                raise ValueError(
                    f"Unknown gate: {gate!r}. Expected 'rezero', 'identity', or an nn.Module."
                )

        match covariance_factor:
            case nn.Module():
                self.covariance_factor = covariance_factor

            case "constant":
                self.covariance_factor = CholeskyFactor(m)

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

        match noise:
            case "scalar":
                self.noise_cholesky = PositiveScalarMatrix(n)
            case "diagonal":
                self.noise_cholesky = PositiveDiagonalMatrix(n)
            case _:
                raise ValueError(
                    f"Unknown noise: {noise!r}. Expected 'scalar' or 'diagonal'."
                )

    # @signature("[(..., n), (..., m)] -> (..., m)")
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
        J = self.noise_cholesky().expand(*batch_shape, self.input_size, self.input_size)
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

    .. math:: Lᵢⱼ(x) =
        \begin{cases}
            \frac{ϕᵢ(x)ᵀϕⱼ(x)}{\sqrt{dₐ}}, & i > j, \\
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
