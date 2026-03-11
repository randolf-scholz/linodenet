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
    "wide_angle_sphere_init",
    "sample_unique_sequences",
    "thomson_initialization",
    "OptimizerResult",
    "OptimizerStatus",
]

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.random import Generator, default_rng
from scipy.stats import ortho_group  # Haar-random orthogonal matrix


def sample_unique_sequences(
    size: int,
    seq_length: int,
    num_samples: int,
    *,
    rng: Optional[int | Generator] = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """Sample num_samples unique sequences of length seq_length from a set of items."""
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
    """Apply a random rotation to the points to avoid axis-alignment."""
    d = points.shape[-1]
    U = ortho_group.rvs(d, random_state=rng)
    return points @ U


def wide_angle_sphere_init(
    n: int,
    d: int,
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

    if n <= 0 or d <= 0:
        raise ValueError("n and d must be positive integers.")
    if n < 1:
        raise ValueError("n must be >= 1.")
    if d == 1:
        # sample from {-1, +1} with equal probability, which is optimal for d=1
        points = rng.choice([-1, 1], size=(n, 1))
        return points.astype(dtype)

    if n == 1:
        # for n=1, we can just return any point on the sphere, e.g. the north-pole
        p = np.eye(1, d, dtype=dtype)
        return _randomize_orientation(p, rng=rng)

    if d == 2:
        # for d=2, we can just put the points uniformly on the circle, which is optimal
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False, dtype=dtype)
        p = np.stack([jnp.cos(angles), np.sin(angles)], axis=-1)
        return _randomize_orientation(p, rng=rng)

    λ = n
    k = math.ceil((λ * n / d) ** (1 / (d - 1)))
    n_faces = 2 * d
    # step 1: assign points to faces of the hypercube
    face = np.array(rng.choice(n_faces, n))
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
    result = np.empty((n, d), dtype=dtype)

    # step 3: for each face, sample points from the grid and insert ±1 at the appropriate position
    for i, n_face in enumerate(points_per_face):
        # step 3a: sample codes of length d-1 without replacement from the grid L
        codes = sample_unique_sequences(k, d - 1, n_face, rng=rng)
        values = L[codes]  # (n_face, d-1)
        # step 3b: insert ±1 at the i-th position to get points on the face (even: +1, odd: -1)
        sign = 1 - 2 * (i % 2)  # +1 for even faces, -1 for odd faces
        dim = i // 2  # dimension of the face normal
        values = np.insert(values, dim, sign, axis=1)
        result[face == i] = values

    # step 4: center and radially project the points onto the sphere
    result = result - np.mean(result, axis=0, keepdims=True)  # safe with n≥2 points.
    result /= np.linalg.norm(result, axis=1, keepdims=True)

    # # ensure points are on the sphere
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)

    # finally, apply a random rotation to avoid axis-alignment
    return _randomize_orientation(result, rng=rng)


def loss_sep(X: Array, beta: float) -> Array:
    r"""Separation loss for Thomson initialization.

    ℓᵦ(X) = softmaxᵦ(XXᵀ - 2𝕀ₙ)
    """
    mask = jnp.eye(X.shape[0], dtype=bool)
    Z = jnp.where(mask, -jnp.inf, X @ X.T)
    log_beta = jnp.log(beta)
    return jax.nn.logmeanexp(log_beta * Z) / beta


def loss_center(X: Array) -> Array:
    r"""Center loss for Thomson initialization (½‖1/n∑ₙxₙ‖²)."""
    center = jnp.mean(X, axis=0)
    return 0.5 * jnp.dot(center, center)


def total_loss(X: Array, beta: float = 5.0, mu: float = 0.1) -> Array:
    """Total loss for Thomson initialization.

    Args:
        X: (n, d) array of n points on the sphere.
        beta: temperature for the separation loss (larger beta = more like max).
        mu: weight for the center loss.

    𝓛(X) = ℓᵦ(X) + μ * ½‖1/n∑ₙxₙ‖²
    """
    l1 = loss_sep(X, beta)
    l2 = loss_center(X)
    return l1 + mu * l2


def n_sphere_geodesic(t: Array, start: Array, direction: Array) -> Array:
    r"""Exponential map on the sphere.

    Assumes both x and g are normalized, i.e. ‖x‖=1 and ‖g‖=1.
    """
    t = t.reshape(-1, 1)
    return jnp.cos(t) * start + jnp.sin(t) * direction


PHI = (1.0 + jnp.sqrt(5.0)) / 2.0
INV_PHI = 1 / PHI


def golden_section_search(
    fn, lower: Array | float, upper: Array | float, maxiter: int = 20
) -> Array:
    r"""Jax implementation of bounded line search via golden-section search."""
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)

    for _ in range(maxiter):
        delta = (upper - lower) * INV_PHI
        c = upper - delta
        d = lower + delta
        f_c = fn(c)
        f_d = fn(d)
        upper = jnp.where(f_c < f_d, d, upper)
        lower = jnp.where(f_c < f_d, lower, c)

    return (lower + upper) / 2


def bisection_line_search(
    fn, lower: Array | float, upper: Array | float, maxiter: int = 20
) -> Array:
    r"""Jax implementation of bounded line search via bisection method."""
    grad_fn = jax.grad(fn)
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)
    for _ in range(maxiter):
        mid = (lower + upper) / 2
        g_mid = grad_fn(mid)
        go_right = g_mid < 0
        lower = jnp.where(go_right, mid, lower)
        upper = jnp.where(go_right, upper, mid)
    return (lower + upper) / 2


def backtracking_line_search(
    fn,
    lower: Array | float,
    upper: Array | float,
    rho: float = 0.5,
    c: float = 1e-3,
) -> Array:
    r"""Jax implementation of backtracking line search (Armijo condition)."""
    vgf = jax.value_and_grad(fn)
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)

    y0, g0 = vgf(lower)
    p = upper - lower  # search direction
    m = jnp.linalg.vecdot(g0, p)

    alpha = 1.0  # start with a full step
    x = lower + alpha * p
    mask = fn(x) > y0 + c * alpha * m

    while mask.any():
        # reduce the step size
        alpha = rho * alpha
        x = jnp.where(mask, lower + alpha * p, x)
        mask = fn(x) > y0 + c * alpha * m

    return x


def safe_bisection_line_search(
    fn, lower: Array | float, upper: Array | float, maxiter: int = 20
) -> Array:
    r"""Bounded line search for gradient descent using left exploration + bisection.

    Assumes phi'(lower) < 0 (descent at the left endpoint).
    Returns a point in [lower, upper] with fn(t) < fn(lower) among sampled points.
    """
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)

    # step 1: backtracking line search to find a reference point
    # with a lower function value than the left endpoint
    x_backtrack = backtracking_line_search(fn, lower, upper)

    # Phase 2: derivative bisection (exploitation)
    x_bisect = bisection_line_search(fn, lower, x_backtrack, maxiter=maxiter)

    return jnp.where(fn(x_backtrack) < fn(x_bisect), x_backtrack, x_bisect)


def sphere_gradient(x: Array, g_euclidean: Array) -> Array:
    """Project the Euclidean gradient onto the tangent space of the sphere."""
    return g_euclidean - jnp.expand_dims(jnp.vecdot(g_euclidean, x), -1) * x


def step(x: Array, loss_fn, grad_fn) -> Array:
    # Negative Euclidean gradient by autodiff
    g_euclidean = -grad_fn(x)
    # Convert to Riemannian gradient on tangent space of the sphere
    g = sphere_gradient(x, g_euclidean)

    # normalize the gradient to get a unit direction on the sphere
    g_norm = jnp.linalg.norm(g, axis=1, keepdims=True)
    g = g / (g_norm + 1e-8)

    # apply retraction (exponential map) to update the points on the sphere
    # For the sphere, the geodesic from x in the direction of g is:
    # γ(t) = cos(‖g‖t)x + sin(‖g‖t)g/‖g‖
    # assuming g is normalized, this is
    # γ(t) = cos(t)x + sin(t)g
    def fn(_t: Array, /) -> Array:
        # loss along the geodesic at time t
        return loss_fn(n_sphere_geodesic(_t, x, g))

    # line search to find the optimal step size along the geodesic
    lowers = jnp.zeros(x.shape[0])
    uppers = jnp.full(x.shape[0], jnp.pi / 4)
    t = safe_bisection_line_search(fn, lowers, uppers, maxiter=10)
    x = n_sphere_geodesic(t, start=x, direction=g)

    # normalize
    x = x / jnp.linalg.norm(x, axis=1, keepdims=True)
    return x


class OptimizerStatus(IntEnum):
    """Status of the optimization."""

    UNKNOWN = -1
    SUCCESS = 0
    NO_CONVERGENCE = 1
    CONSTRAINT_VIOLATION = 2
    NON_FINITE_VALUES = 3


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class OptimizerResult:
    """Result of the optimization."""

    x: Array
    fun: Array | float
    jac: Array
    success: bool
    nit: int | None = None
    maxcv: Array | float | None = None
    status: OptimizerStatus = OptimizerStatus.UNKNOWN
    loss_hist: list[float] = field(default_factory=list)
    grad_hist: list[float] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)


def scaled_norm(x: Array, axis: None | int | tuple[int, ...] = None) -> Array:
    """Computes $√(1/n ∑ₙ xₙ²)$."""
    # convert to tuple
    match axis:
        case None:
            axis = tuple(range(x.ndim))
        case int():
            axis = (axis % x.ndim,)
        case tuple():
            axis = tuple(a % x.ndim for a in axis)
        case _:
            raise ValueError(f"Invalid axis: {axis!r}")

    batch = tuple(a for a in range(x.ndim) if a not in axis)
    dims = ((axis, axis), (batch, batch))

    n = math.prod(x.shape[a] for a in axis)
    return jnp.sqrt(jax.lax.dot(x, x, dimension_numbers=dims) / n)


def thomson_initialization(
    n: int,
    d: int,
    *,
    beta=5.0,
    maxiter=100,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    patience: int = 5,
    seed: Optional[int | Generator] = None,
) -> OptimizerResult:
    x = jnp.array(wide_angle_sphere_init(n, d, seed=seed))
    if n == 1 or d == 1:
        return OptimizerResult(
            x=x,
            fun=0.0,
            jac=jnp.zeros_like(x),
            success=True,
            options={
                "beta": beta,
                "atol": atol,
                "rtol": rtol,
                "maxiter": maxiter,
            },
        )

    loss_fn = jax.tree_util.Partial(loss_sep, beta=beta)
    grad_fn = jax.grad(loss_fn)
    val_and_grad_fn = jax.value_and_grad(loss_fn)

    loss, grad = val_and_grad_fn(x)
    grad = sphere_gradient(x, grad)
    grad_norm = scaled_norm(grad, axis=-1).sum()

    # initialize the history lists with the initial loss and gradient norm
    loss_hist: list[float] = [float(loss.item())]
    grad_hist: list[float] = [float(grad_norm.item())]

    it = 0
    for it in range(maxiter):
        x = step(x, loss_fn, grad_fn)
        loss, grad = val_and_grad_fn(x)
        grad = sphere_gradient(x, grad)
        grad_norm = scaled_norm(grad, axis=-1).sum()
        loss_hist.append(float(loss.item()))
        grad_hist.append(float(grad_norm.item()))

        if not jnp.isfinite(loss) or not jnp.isfinite(grad_norm):
            print(
                f"Warning: Non-finite value encountered at iteration {it}."
                f" loss={loss:.6f}, grad_norm={grad_norm:.6f}"
            )
            status = OptimizerStatus.NON_FINITE_VALUES
            break

        if not jnp.allclose(jnp.linalg.norm(x, axis=-1), 1.0, atol=atol, rtol=rtol):
            print(
                f"Warning: Constraint violation at iteration {it}."
                f" Points not on the sphere: max deviation={jnp.max(jnp.abs(jnp.linalg.norm(x, axis=-1) - 1)):.6f}"
            )
            status = OptimizerStatus.CONSTRAINT_VIOLATION
            break

        m = -min(patience + 1, it + 2)  # look back at most p iterations
        if (grad_hist[m] - grad_norm) < rtol * grad_hist[m] + atol:
            print(
                f"Converged after {it} iterations."
                f" Final loss: {loss:.6f}"
                f" Final (per particle) grad norm: {grad_norm:.6f}"
            )
            status = OptimizerStatus.SUCCESS
            break

        assert loss < loss_hist[-1] + atol, (
            f"{it=} Loss did not decrease: {loss:.6f} >= {loss_hist[-1]:.6f}"
        )
    else:
        status = OptimizerStatus.NO_CONVERGENCE
        print(
            f"Warning: Did not converge after {maxiter} iterations."
            f" Final loss: {loss:.6f}"
            f" Final (per particle) grad norm: {grad_norm:.6f}"
        )

    maxcv = jnp.abs(jnp.linalg.norm(x, axis=-1) - 1).max()

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
