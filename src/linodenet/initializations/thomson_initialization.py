r"""Thomson initialization for sampling points on the sphere.

The Thomson Problem is a classical problem in physics and mathematics
that asks for the minimum energy configuration of n electrons on the surface of a sphere.
The electrons repel each other with a force inversely proportional to the square of the distance between them,
and the energy of a configuration is given by the sum of the inverse distances between all pairs of electrons.

We generalize this problem to an initialization scheme for sampling points on the surface of a d-dimensional sphere.

That is, we want thomson_init(n, d) to return n points on the surface of a d-dimensional sphere
that are maximally spread out, i.e. the minimum distance between any two points is maximized.

Equivalently, one can ask to maximize the angle between any two points on the sphere,
which is itself equivalent to minimizing the inner product.

For n-many points, collected in a matrix $X∈ℝ^{n×d}$, their pair wise inner products
are the entries of the n×n Gram matrix $G=XX^T$,
and the sum of the inner products is given by the sum of the entries of G.
"""

__all__ = [
    "OptimizerResult",
    "OptimizerStatus",
    # functions
    "wide_angle_sphere_init",
    "thomson_initialization",
]

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from functools import partial
from typing import Any, Optional

import numpy as np
import torch
from numpy.random import Generator, default_rng
from scipy.stats import ortho_group  # Haar-random orthogonal matrix
from torch import Tensor


def _sample_unique_sequences(
    size: int,
    seq_length: int,
    num_samples: int,
    *,
    rng: Optional[int | Generator] = None,
    batch_size: int = 4096,
) -> np.ndarray:
    r"""Sample num_samples unique sequences of length seq_length from a set of items."""
    if num_samples > size**seq_length:
        raise ValueError("num_samples must be less than or equal to size^seq_length.")

    dtype: type[np.unsignedinteger]
    if size < np.iinfo(np.uint8).max:
        dtype = np.uint8
    elif size < np.iinfo(np.uint16).max:
        dtype = np.uint16
    elif size < np.iinfo(np.uint32).max:
        dtype = np.uint32
    elif size < np.iinfo(np.uint64).max:
        dtype = np.uint64
    else:
        raise ValueError("size must be less than or equal to 2^64.")

    # Output indices into A; map to A at the end
    out_idx = np.empty((num_samples, seq_length), dtype=dtype)
    seen: set[bytes] = set()
    filled = 0
    rng = default_rng(rng)

    while filled < num_samples:
        m = min(batch_size, num_samples - filled)

        # Sample candidate index rows uniformly from {0, ..., k-1}^d
        cand = rng.integers(0, size, (m, seq_length), dtype=dtype)

        # remove duplicates from the batch
        cand = np.unique(cand, axis=0)

        mask = np.array([row.tobytes() in seen for row in cand])
        cand_new = cand[~mask]
        new = len(cand_new)
        out_idx[filled : filled + new] = cand_new
        filled += new

    return out_idx


def _randomize_orientation(points: np.ndarray, rng: Generator) -> np.ndarray:
    r"""Apply a random rotation to the points to avoid axis-alignment."""
    d = points.shape[-1]
    U = ortho_group.rvs(d, random_state=rng)
    return points @ U


def wide_angle_sphere_init(
    num: int,
    dim: int,
    *,
    dtype=np.float32,
    seed: Optional[int | Generator] = None,
    uniform_grid: bool = False,
) -> np.ndarray:
    """Initialize n points on Sᵈ⁻¹ with some guaranteed separation.

    We wrap the hypersphere into a hypercube.
    Each face of the hypercube is itself a full (d-1) - hypercube
    We put a grid G=Lᵈ⁻¹ with |L|=k points on each face,
    sample points from this grid, and project them radially.

    Since there are 2d faces, there are p=2dkᵈ⁻¹ points to pick from.
    We choose k such that the total number of points is large compared to n,
    that is p≥λn, which gives k⁎=⌈(λn/(2d))^(1/(d-1))⌉

    As we will use rejection sampling, we pick lambda to ensure a small number of rejections
    The expected number of collisions is n(n-1)/2p,
    so we want p to be large compared to n², which gives λ≥n.
    Using λ=r⋅n, we have k⁎=⌈(rn²/(2d))^(1/(d-1))⌉, p=2dk⁎ᵈ⁻¹≥rn²,
    and the expected number of collisions is n(n-1)/(rn²)≈1/r.`
    So even with r=1, we expect only a constant number of rejections.
    We pick r=2 to be safe, which gives k⁎=⌈(n²/d)^(1/(d-1))⌉ and p≥2n².
    """
    rng = default_rng(seed)
    del seed  # avoid accidentally using the seed as a random state later on

    if num <= 0 or dim <= 0:
        raise ValueError("n and d must be positive integers.")
    if num < 1:
        raise ValueError("n must be >= 1.")
    if dim == 1:
        # sample from {-1, +1} with equal probability, which is optimal for d=1
        points = rng.choice([-1, 1], size=(num, 1))
        return points.astype(dtype)

    if num == 1:
        # for n=1, we can just return any point on the sphere, e.g. the north-pole
        p = np.eye(1, dim, dtype=dtype)
        return _randomize_orientation(p, rng=rng)

    if dim == 2:
        # for d=2, we can just put the points uniformly on the circle, which is optimal
        angles = np.linspace(0, 2 * math.pi, num, endpoint=False, dtype=dtype)
        p = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        return _randomize_orientation(p, rng=rng)

    λ = num
    k = math.ceil((λ * num / dim) ** (1 / (dim - 1)))
    n_faces = 2 * dim
    # step 1: assign points to faces of the hypercube
    face = np.array(rng.choice(n_faces, num))
    points_per_face = np.bincount(face, minlength=n_faces)
    # points_per_face = np.full(n_faces, n // n_faces, dtype=np.int32)
    # points_per_face[-1] += n % n_faces
    # face = np.repeat(np.arange(n_faces), points_per_face)

    # step 2: set up the 1-d grid on each face, excluding the end points to avoid duplicates across faces
    if uniform_grid:
        # (a) uniform grid in [-1, +1], excluding endpoints:
        #     k=0: {0}
        #     k=1: {-½, +½}
        #     k=2: {-⅓, 0, +⅓}
        #     k=3: {-¾, -¼, +¼, +¾}
        L = 2 * ((np.arange(k) - 1) / (k + 1)) - 1  # (k_star,)
    else:
        # (b) non-linear grid that is denser near the center
        L = (2 * np.arange(k) + 1 - k) / k
        L = np.tan(L * np.pi / 4)

    # step 0: pre-allocate the output array
    result = np.empty((num, dim), dtype=dtype)

    # step 3: for each face, sample points from the grid and insert ±1 at the appropriate position
    for i, n_face in enumerate(points_per_face):
        # step 3a: sample codes of length d-1 without replacement from the grid L
        codes = _sample_unique_sequences(k, dim - 1, n_face, rng=rng)
        values = L[codes]  # (n_face, d-1)
        # step 3b: insert ±1 at the i-th position to get points on the face (even: +1, odd: -1)
        sign = 1 - 2 * (i % 2)  # +1 for even faces, -1 for odd faces
        pos = i // 2  # dimension of the face normal
        values = np.insert(values, pos, sign, axis=1)
        result[face == i] = values

    # step 4: center and radially project the points onto the sphere
    result = result - np.mean(result, axis=0, keepdims=True)  # safe with n≥2 points.
    result /= np.linalg.norm(result, axis=1, keepdims=True)

    # ensure points are on the sphere
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)

    # finally, apply a random rotation to avoid axis-alignment
    return _randomize_orientation(result, rng=rng)


def logmeanexp(x: Tensor, /) -> Tensor:
    r"""Log-mean-exp over all elements."""
    flat = x.reshape(-1)
    return torch.logsumexp(flat, dim=0) - math.log(flat.numel())


def loss_sep(x: Tensor, /, *, beta: float | Tensor) -> Tensor:
    r"""Separation loss for Thomson initialization.

    .. math:: ℓᵦ(X) = \softmaxᵦ(XXᵀ - 2𝕀ₙ) ≈ \max_{i≠j} ⟨xᵢ∣xⱼ⟩
    """
    mask = torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
    Z = x @ x.T
    Z = Z.masked_fill(mask, -torch.inf)
    beta_t = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
    log_beta = torch.log(beta_t)
    return logmeanexp(log_beta * Z) / beta_t


def loss_center(x: Tensor, /) -> Tensor:
    r"""Center loss for Thomson initialization (½‖1/n∑ₙxₙ‖²)."""
    center = torch.mean(x, dim=0)
    return 0.5 * torch.dot(center, center)


def total_loss(x: Tensor, /, *, beta: float = 5.0, mu: float = 0.1) -> Tensor:
    """Total loss for Thomson initialization.

    Args:
        x: (n, d) array of n points on the sphere.
        beta: temperature for the separation loss (larger beta = more like max).
        mu: weight for the center loss.

    𝓛(X) = ℓᵦ(X) + μ * ½‖1/n∑ₙxₙ‖²
    """
    l1 = loss_sep(x, beta=beta)
    l2 = loss_center(x)
    return l1 + mu * l2


def n_sphere_geodesic(t: Tensor, /, start: Tensor, direction: Tensor) -> Tensor:
    r"""Exponential map on the sphere.

    Assumes both x and g are normalized, i.e. ‖x‖=1 and ‖g‖=1.
    """
    t = t.reshape(-1, 1)
    return torch.cos(t) * start + torch.sin(t) * direction


PHI = (1.0 + math.sqrt(5.0)) / 2.0
INV_PHI = 1 / PHI


def golden_section_search(
    fn: Callable[[Tensor], Tensor],
    /,
    lower: Tensor | float,
    upper: Tensor | float,
    *,
    maxiter: int = 20,
) -> Tensor:
    r"""Torch implementation of bounded line search via golden-section search."""
    lower = torch.as_tensor(lower)
    upper = torch.as_tensor(upper)

    for _ in range(maxiter):
        delta = (upper - lower) * INV_PHI
        c = upper - delta
        d = lower + delta
        f_c = fn(c)
        f_d = fn(d)
        upper = torch.where(f_c < f_d, d, upper)
        lower = torch.where(f_c < f_d, lower, c)

    return (lower + upper) / 2


def bisection_line_search(
    fn: Callable[[Tensor], Tensor],
    /,
    lower: Tensor | float,
    upper: Tensor | float,
    *,
    maxiter: int = 20,
) -> Tensor:
    r"""Torch implementation of bounded line search via bisection method."""
    grad_fn = torch.func.grad(fn)
    lower = torch.as_tensor(lower)
    upper = torch.as_tensor(upper)
    for _ in range(maxiter):
        mid = (lower + upper) / 2
        g_mid = grad_fn(mid)
        go_right = g_mid < 0
        lower = torch.where(go_right, mid, lower)
        upper = torch.where(go_right, upper, mid)
    return (lower + upper) / 2


def backtracking_line_search(
    fn: Callable[[Tensor], Tensor],
    /,
    lower: Tensor | float,
    upper: Tensor | float,
    *,
    rho: float = 0.5,
    c: float = 1e-3,
) -> Tensor:
    r"""Torch implementation of backtracking line search (Armijo condition)."""
    grad_and_value_fn = torch.func.grad_and_value(fn)
    lower = torch.as_tensor(lower)
    upper = torch.as_tensor(upper)

    g0, y0 = grad_and_value_fn(lower)
    p = upper - lower  # search direction
    m = torch.linalg.vecdot(g0, p)

    alpha = 1.0  # start with a full step
    x = lower + alpha * p
    mask = fn(x) > y0 + c * alpha * m

    while mask.any().item():
        # reduce the step size
        alpha = rho * alpha
        x = torch.where(mask, lower + alpha * p, x)
        mask = fn(x) > y0 + c * alpha * m

    return x


def safe_bisection_line_search(
    fn: Callable[[Tensor], Tensor],
    /,
    lower: Tensor | float,
    upper: Tensor | float,
    *,
    maxiter: int = 20,
) -> Tensor:
    r"""Bounded line search for gradient descent using left exploration + bisection.

    Assumes phi'(lower) < 0 (descent at the left endpoint).
    Returns a point in [lower, upper] with fn(t) < fn(lower) among sampled points.
    """
    lower = torch.as_tensor(lower)
    upper = torch.as_tensor(upper)

    # step 1: backtracking line search to find a reference point
    # with a lower function value than the left endpoint
    x_backtrack = backtracking_line_search(fn, lower, upper)

    # Phase 2: derivative bisection (exploitation)
    x_bisect = bisection_line_search(fn, lower, x_backtrack, maxiter=maxiter)

    return torch.where(fn(x_backtrack) < fn(x_bisect), x_backtrack, x_bisect)


def sphere_gradient(x: Tensor, /, *, grad_euclidean: Tensor) -> Tensor:
    r"""Project the Euclidean gradient onto the tangent space of the sphere."""
    dot = torch.linalg.vecdot(grad_euclidean, x).unsqueeze(-1)
    return grad_euclidean - dot * x


def geodesic_loss(
    t: Tensor,
    /,
    *,
    point: Tensor,
    grad: Tensor,
    loss_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    r"""Loss along the geodesic at time t."""
    return loss_fn(n_sphere_geodesic(t, point, grad))


def step(
    x: Tensor,
    /,
    *,
    loss_fn: Callable[[Tensor], Tensor],
    grad_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    # Negative Euclidean gradient by autodiff
    g_euclidean = -grad_fn(x)
    # Convert to Riemannian gradient on tangent space of the sphere
    g = sphere_gradient(x, grad_euclidean=g_euclidean)

    # normalize the gradient to get a unit direction on the sphere
    g_norm = torch.linalg.norm(g, dim=1, keepdim=True)
    g = g / (g_norm + 1e-8)

    # apply retraction (exponential map) to update the points on the sphere
    # For the sphere, the geodesic from x in the direction of g is:
    # γ(t) = cos(‖g‖t)x + sin(‖g‖t)g/‖g‖
    # assuming g is normalized, this is
    # γ(t) = cos(t)x + sin(t)g
    fn = partial(geodesic_loss, point=x, grad=g, loss_fn=loss_fn)

    # line search to find the optimal step size along the geodesic
    lowers = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    uppers = torch.full((x.shape[0],), math.pi / 4, dtype=x.dtype, device=x.device)
    t = safe_bisection_line_search(fn, lowers, uppers, maxiter=10)
    x = n_sphere_geodesic(t, start=x, direction=g)

    # normalize
    x = x / torch.linalg.norm(x, dim=1, keepdim=True)
    return x


class OptimizerStatus(IntEnum):
    r"""Status of the optimization."""

    UNKNOWN = -1
    SUCCESS = 0
    NO_CONVERGENCE = 1
    CONSTRAINT_VIOLATION = 2
    NON_FINITE_VALUES = 3


@dataclass(frozen=True, slots=True)
class OptimizerResult:
    r"""Result of the optimization."""

    x: Tensor
    fun: Tensor | float
    jac: Tensor
    success: bool
    nit: int | None = None
    maxcv: Tensor | float | None = None
    status: OptimizerStatus = OptimizerStatus.UNKNOWN
    loss_hist: list[float] = field(default_factory=list)
    grad_hist: list[float] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)


def scaled_norm(x: Tensor, axis: None | int | tuple[int, ...] = None) -> Tensor:
    r"""Computes $√(1/n ∑ₙ xₙ²)$."""
    # convert to tuple
    match axis:
        case None:
            return torch.sqrt(torch.mean(x.pow(2)))
        case int():
            axis = (axis % x.ndim,)
        case tuple():
            axis = tuple(a % x.ndim for a in axis)
        case _:
            raise ValueError(f"Invalid axis: {axis!r}")
    return torch.sqrt(torch.mean(x.pow(2), dim=axis))


def thomson_initialization(
    num: int,
    dim: int,
    *,
    beta: float = 5.0,
    maxiter: int = 100,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    patience: int = 5,
    seed: Optional[int | Generator] = None,
) -> OptimizerResult:
    r"""Thomson initialization for sampling n points on the surface of a d-dimensional sphere.

    The Thomson problem asks for the minimum energy configuration of n electrons on the surface of a sphere,
    where the energy is given by the sum of the inverse distances between all pairs of electrons.
    We generalize this problem to an initialization scheme for sampling points on the surface of a d-dimensional sphere.

    We optimize the following loss function:

    .. math:: 𝓛 &= \softmaxᵦ(XXᵀ - 2𝕀ₙ) ≈ \max_{i≠j} ⟨xᵢ∣xⱼ⟩

    Args:
        num: number of points to sample.
        dim: dimension of the sphere (i.e. points are in ℝᵈ).
        beta: temperature for the separation loss (larger beta = more like max).
        maxiter: maximum number of iterations for the optimization.
        atol: absolute tolerance for convergence and constraint violation.
        rtol: relative tolerance for convergence and constraint violation.
        patience: number of iterations to look back for convergence check.
        seed: random seed for initialization.

    Returns:
        OptimizerResult
    """
    x = torch.as_tensor(wide_angle_sphere_init(num, dim, seed=seed))
    if num == 1 or dim == 1:
        return OptimizerResult(
            x=x,
            fun=0.0,
            jac=torch.zeros_like(x),
            success=True,
            options={
                "beta": beta,
                "atol": atol,
                "rtol": rtol,
                "maxiter": maxiter,
            },
        )

    loss_fn = partial(loss_sep, beta=beta)
    grad_fn = torch.func.grad(loss_fn)
    grad_and_value_fn = torch.func.grad_and_value(loss_fn)

    grad, loss = grad_and_value_fn(x)
    grad = sphere_gradient(x, grad_euclidean=grad)
    grad_norm = scaled_norm(grad, axis=-1).sum()
    loss_value = float(loss)
    grad_norm_value = float(grad_norm)

    # initialize the history lists with the initial loss and gradient norm
    loss_hist: list[float] = [float(loss)]
    grad_hist: list[float] = [float(grad_norm)]

    it = 0
    for it in range(maxiter):
        x = step(x, loss_fn=loss_fn, grad_fn=grad_fn)
        grad, loss = grad_and_value_fn(x)
        grad = sphere_gradient(x, grad_euclidean=grad)
        grad_norm = scaled_norm(grad, axis=-1).sum()
        loss_value = float(loss)
        grad_norm_value = float(grad_norm)
        loss_hist.append(loss_value)
        grad_hist.append(grad_norm_value)

        if not torch.isfinite(loss).item() or not torch.isfinite(grad_norm).item():
            print(
                f"Warning: Non-finite value encountered at iteration {it}."
                f" loss={loss_value:.6f}, grad_norm={grad_norm_value:.6f}"
            )
            status = OptimizerStatus.NON_FINITE_VALUES
            break

        if not torch.allclose(
            torch.linalg.norm(x, dim=-1),
            torch.ones(x.shape[0], dtype=x.dtype, device=x.device),
            atol=atol,
            rtol=rtol,
        ):
            max_deviation = torch.abs(torch.linalg.norm(x, dim=-1) - 1).max().item()
            print(
                f"Warning: Constraint violation at iteration {it}."
                f" Points not on the sphere: {max_deviation=:.6f}"
            )
            status = OptimizerStatus.CONSTRAINT_VIOLATION
            break

        m = -min(patience + 1, it + 2)  # look back at most p iterations
        if (grad_hist[m] - grad_norm_value) < rtol * grad_hist[m] + atol:
            print(
                f"Converged after {it} iterations."
                f" Final loss: {loss_value:.6f}"
                f" Final (per particle) grad norm: {grad_norm_value:.6f}"
            )
            status = OptimizerStatus.SUCCESS
            break

        assert loss_value < loss_hist[-1] + atol, (
            f"{it=} Loss did not decrease: {loss_value:.6f} >= {loss_hist[-1]:.6f}"
        )
    else:
        status = OptimizerStatus.NO_CONVERGENCE
        print(
            f"Warning: Did not converge after {maxiter} iterations."
            f" Final loss: {loss_value:.6f}"
            f" Final (per particle) grad norm: {grad_norm_value:.6f}"
        )

    maxcv = torch.abs(torch.linalg.norm(x, dim=-1) - 1).max()

    result = OptimizerResult(
        x=x,
        fun=loss,
        jac=grad,
        success=status is OptimizerStatus.SUCCESS,
        status=status,
        nit=it,
        loss_hist=loss_hist,
        grad_hist=grad_hist,
        maxcv=maxcv,
        options={
            "beta": beta,
            "atol": atol,
            "rtol": rtol,
            "maxiter": maxiter,
        },
    )

    return result
