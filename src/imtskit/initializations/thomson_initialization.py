r"""Thomson initialization for sampling points on the sphere.

The classical Thomson problem asks for the minimum-energy configuration of
electrons on the surface of a sphere. Its energy is the sum of inverse distances between
distinct pairs of electrons.

Here, `thomson_initialization` seeks well-separated points on the unit sphere
$Sᵈ⁻¹⊂ℝᵈ$ using an inner-product loss. Maximizing the minimum pairwise distance
is equivalent to maximizing the minimum angle, or minimizing the maximum
off-diagonal inner product. This objective differs from the classical Thomson energy.

For $n$ points collected as rows of $X∈ℝ^{n×d}$, pairwise inner products are
the entries of the $n×n$ Gram matrix $G=XXᵀ$. `separation_loss` aggregates its
off-diagonal entries, and `wide_angle_sphere_init` provides starting points.
"""

__all__ = [
    "OptimizerResult",
    "OptimizerStatus",
    # extras
    "backtracking_line_search",
    "bisection_line_search",
    "golden_section_search",
    "safe_bisection_line_search",
    "separation_loss",
    "sphere_geodesic",
    "sphere_gradient",
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

import torch
from torch import Tensor
from torch.linalg import vecdot, vector_norm

from signatures import signature


def _sample_unique_sequences(
    size: int,
    seq_length: int,
    num_samples: int,
    *,
    device: Optional[str | torch.device] = None,
    batch_size: int = 4096,
) -> Tensor:
    r"""Sample `num_samples` distinct sequences from `range(size)`.

    Each sequence has length `seq_length`; entries within a sequence may repeat.
    Requires `num_samples ≤ size**seq_length` and `size ≤ 2**63 - 1`.
    """
    if num_samples > size**seq_length:
        raise ValueError("num_samples must be less than or equal to size^seq_length.")
    if size > 2**63 - 1:
        raise ValueError("size must be less than or equal to 2^63 - 1.")

    out_idx = torch.empty((num_samples, seq_length), dtype=torch.int64, device=device)
    seen: set[tuple[int, ...]] = set()
    filled = 0

    while filled < num_samples:
        m = min(batch_size, num_samples - filled)
        cand = torch.randint(0, size, (m, seq_length), device=device)
        cand = torch.unique(cand, dim=-2)

        for row in cand:
            key = tuple(row.tolist())
            if key in seen:
                continue
            seen.add(key)
            out_idx[filled] = row
            filled += 1
            if filled == num_samples:
                break

    return out_idx


def _random_orthogonal_matrix(
    dim: int,
    /,
    *,
    dtype: torch.dtype,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    r"""Sample a Haar-distributed orthogonal matrix via QR."""
    A = torch.randn(dim, dim, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(A)
    d = torch.diagonal(R)
    signs = torch.where(d == 0, 1.0, d.sign())
    return Q * signs.unsqueeze(0)


def _randomize_orientation(
    points: Tensor,
    /,
) -> Tensor:
    r"""Apply a random orthogonal transformation and renormalize the points."""
    U = _random_orthogonal_matrix(
        points.shape[-1],
        dtype=points.dtype,
        device=points.device,
    )
    points = points @ U
    # renormalize (should be close to 1 still, but just to be safe)
    points = points / vector_norm(points, dim=-1, keepdim=True)
    return points


def wide_angle_sphere_init(
    num: int,
    dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
    seed: Optional[int] = None,
    uniform_grid: bool = False,
) -> Tensor:
    r"""Initialize $n$ points on the unit sphere $Sᵈ⁻¹⊂ℝᵈ$ from coordinate grids.

    For $d>1$ and $n>1$, assign points to the $2d$ coordinate faces of a
    hypercube. Each face has a Cartesian grid with $k$ coordinates per axis
    and $kᵈ⁻¹$ candidate points. Sample distinct grid codes within each face,
    project the points radially onto the sphere, and randomize their orientation.

    Choose the grid size so the total number of candidates $p$ is at least $2n²$:

    $$  k = ⌈(n²/d)^{1/(d-1)}⌉, \qquad p = 2dkᵈ⁻¹ ≥ 2n²  $$

    This construction does not optimize separation.

    Note: Derivation of the grid size

        The $2d$ faces each supply $kᵈ⁻¹$ candidates, giving $p=2dkᵈ⁻¹$.
        For an oversampling factor $λ≥1$, require $p≥λn$. Solving for the
        smallest integer grid size gives:

        $$  2dkᵈ⁻¹ ≥ λn \quad ⇒ \quad k⁎ = ⌈(λn/(2d))^{1/(d-1)}⌉  $$

        For $n$ independent uniform draws from $p$ candidates, each pair
        collides with probability $1/p$, so the expected number of colliding
        pairs is $n(n-1)/(2p)$. To keep this estimate bounded as $n$ grows,
        choose $λ=rn$ with a constant $r≥1$. Then:

        $$
            k⁎ = ⌈(rn²/(2d))^{1/(d-1)}⌉,
            \qquad p ≥ rn²,
            \qquad \frac{n(n-1)}{2p} ≤ \frac{n-1}{2rn} < \frac{1}{2r}
        $$

        Thus even $r=1$ gives a constant collision bound. We use $r=2$,
        yielding $k=k⁎=⌈(n²/d)^{1/(d-1)}⌉$ and $p≥2n²$, with an expected
        number of colliding pairs at most $(n-1)/(4n)<¼$. This estimate
        motivates the grid size for rejection sampling within each face.

    Args:
        num: Number of points $n≥1$.
        dim: Ambient dimension $d≥1$; the sphere has dimension $d-1$.
        dtype: Data type of the returned points.
        device: Device of the returned points.
        seed: Seed for PyTorch's global random number generator, if provided.
        uniform_grid: Use equally spaced grid coordinates instead of the default
            tangent-spaced coordinates.

    Returns:
        Unit vectors with shape `(num, dim)`. For `dim=1`, points are sampled
        from $\{-1,+1\}$ with replacement, so repetitions are possible.

    Raises:
        ValueError: If `num` or `dim` is less than one, or a face is assigned
            more points than its grid can supply.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if dim < 1:
        raise ValueError("dim must be >=1")
    if num < 1:
        raise ValueError("num must be >= 1.")
    if dim == 1:
        # sample from {-1, +1} with equal probability, which is optimal for d=1
        points = 2.0 * torch.randint(0, 2, (num, 1), device=device) - 1.0
        return points.to(dtype=dtype)
    if num == 1:
        # for n=1, we can just return any point on the sphere, e.g. the north-pole
        p = torch.randn(1, dim, device=device, dtype=dtype)
        return p / vector_norm(p, dim=-1)

    # step 1: set up the 1-d grid on each face, excluding the end points to avoid duplicates across faces
    λ = num
    k = math.ceil((λ * num / dim) ** (1 / (dim - 1)))
    grid = torch.arange(k, dtype=dtype, device=device)
    if uniform_grid:
        # (a) uniform grid in [-1, +1], excluding endpoints:
        #     k=0: {0}
        #     k=1: {-½, +½}
        #     k=2: {-⅓, 0, +⅓}
        #     k=3: {-¾, -¼, +¼, +¾}
        grid = 2 * ((grid - 1) / (k + 1)) - 1
    else:
        # (b) non-linear grid that is denser near the center
        grid = (2 * grid + 1 - k) / k
        grid = torch.tan(grid * math.pi / 4)

    # step 2: assign points to faces of the hypercube
    n_faces = 2 * dim
    face = torch.randint(0, n_faces, (num,))
    points_per_face = torch.bincount(face, minlength=n_faces)

    # step 3: for each face, sample points from the grid and insert ±1 at the appropriate position
    # pre-allocate the output array
    result = torch.empty((num, dim), dtype=dtype, device=device)

    for i, n_face in enumerate(points_per_face.tolist()):
        if n_face == 0:
            continue
        # step 3a: sample codes of length d-1 without replacement from the grid L
        codes = _sample_unique_sequences(k, dim - 1, n_face, device=device)
        values = grid[codes]  # (n_face, d-1)
        # step 3b: insert ±1 at the i-th position to get points on the face (even: +1, odd: -1)
        sign = 1 - 2 * (i % 2)  # +1 for even faces, -1 for odd faces
        pos = i // 2  # dimension of the face normal
        values = torch.cat(
            [
                values[:, :pos],
                torch.full((n_face, 1), sign, dtype=dtype, device=device),
                values[:, pos:],
            ],
            dim=-1,
        )
        result[face.to(device=device) == i] = values

    # step 4: radially project the points onto the sphere
    result = result / vector_norm(result, dim=-1, keepdim=True)

    # ensure points are on the sphere
    assert torch.allclose(
        vector_norm(result, dim=-1),
        torch.ones(num, dtype=dtype, device=device),
    )

    # finally, apply a random rotation to avoid axis-alignment
    return _randomize_orientation(result)


@signature("(..., n, d) -> (...)")
def _log_mean_exp(x: Tensor, /) -> Tensor:
    r"""Compute log-mean-exp over the final two axes.

    The normalization uses the total element count, including batch dimensions.
    """
    flat = x.flatten(-2, -1)
    return torch.logsumexp(flat, dim=-1) - math.log(flat.numel())


@signature("(..., n, d) -> (...)")
def separation_loss(x: Tensor, /, *, beta: float | Tensor) -> Tensor:
    r"""Aggregate off-diagonal inner products for Thomson initialization.

    For an unbatched matrix $X∈ℝ^{n×d}$ with unit-length rows $xᵢ$, compute:

    $$  ℓᵦ(X) = \frac{\log(∑_{i≠j} \exp((\log β)⟨xᵢ,xⱼ⟩)) - \log(n²)}{β}  $$

    Diagonal entries of $XXᵀ$ are masked with $-∞$. The current implementation
    scales inner products by $\log β$ and divides the result by $β$;
    this differs from the usual log-mean-exp approximation to the maximum.
    Requires $β>1$ and $n≥2$ for a finite loss.
    """
    mask = torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
    Z = x @ x.T
    Z = Z.masked_fill(mask, -torch.inf)
    beta_t = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
    log_beta = torch.log(beta_t)
    return _log_mean_exp(log_beta * Z) / beta_t


@signature("[(..., n), (..., n, d), (..., n, d)] -> (..., n, d)")
def sphere_geodesic(t: Tensor, /, start: Tensor, direction: Tensor) -> Tensor:
    r"""Follow a unit-speed geodesic on the sphere.

    Requires `start` and `direction` to be unit vectors $x$ and $g$ with $⟨x,g⟩=0$.
    The result $\cos(t)x + \sin(t)g$ then remains on the unit sphere.
    """
    t = t.unsqueeze(-1)
    return torch.cos(t) * start + torch.sin(t) * direction


@signature("{(..., n, d) -> (...)} -> (..., n, d)")
def golden_section_search(
    fn: Callable[[Tensor], Tensor],
    /,
    lower: Tensor | float,
    upper: Tensor | float,
    *,
    maxiter: int = 20,
) -> Tensor:
    r"""Minimize a unimodal function on `[lower, upper]` by golden-section search."""
    INV_PHI = 2.0 / (1.0 + math.sqrt(5.0))  # 1 over golden ratio.

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


@signature("{(..., n, d) -> (...)} -> (..., n, d)")
def bisection_line_search(
    fn: Callable[[Tensor], Tensor],
    /,
    lower: Tensor | float,
    upper: Tensor | float,
    *,
    maxiter: int = 20,
) -> Tensor:
    r"""Locate a minimum by bisecting the derivative on `[lower, upper]`.

    Assumes the derivative is increasing and changes sign across the interval.
    """
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


@signature("{(..., n, d) -> (...)} -> (..., n, d)")
def backtracking_line_search(
    fn: Callable[[Tensor], Tensor],
    /,
    lower: Tensor | float,
    upper: Tensor | float,
    *,
    rho: float = 0.5,
    c: float = 1e-3,
) -> Tensor:
    r"""Find a step from `lower` toward `upper` satisfying the Armijo condition.

    The direction must be a descent direction. `rho` must lie in $(0,1)$
    so successive trial steps shrink.
    """
    grad_and_value_fn = torch.func.grad_and_value(fn)
    lower = torch.as_tensor(lower)
    upper = torch.as_tensor(upper)

    g0, y0 = grad_and_value_fn(lower)
    p = upper - lower  # search direction
    m = vecdot(g0, p, dim=-1)

    alpha = 1.0  # start with a full step
    x = lower + alpha * p
    mask = fn(x) > y0 + c * alpha * m

    while mask.any().item():
        # reduce the step size
        alpha = rho * alpha
        x = torch.where(mask, lower + alpha * p, x)
        mask = fn(x) > y0 + c * alpha * m

    return x


@signature("{(..., n, d) -> (...)} -> (..., n, d)")
def safe_bisection_line_search(
    fn: Callable[[Tensor], Tensor],
    /,
    lower: Tensor | float,
    upper: Tensor | float,
    *,
    maxiter: int = 20,
) -> Tensor:
    r"""Combine backtracking and derivative bisection on `[lower, upper]`.

    Assumes `lower < upper` and $ϕ′(lower)<0$, where $ϕ$ denotes `fn`.
    Return whichever trial point has the smaller function value.
    """
    lower = torch.as_tensor(lower)
    upper = torch.as_tensor(upper)

    # step 1: backtracking line search to find a reference point
    # with a lower function value than the left endpoint
    x_backtrack = backtracking_line_search(fn, lower, upper)

    # Phase 2: derivative bisection (exploitation)
    x_bisect = bisection_line_search(fn, lower, x_backtrack, maxiter=maxiter)

    return torch.where(fn(x_backtrack) < fn(x_bisect), x_backtrack, x_bisect)


@signature("(..., n, d) -> (..., n, d)")
def sphere_gradient(x: Tensor, /, *, grad_euclidean: Tensor) -> Tensor:
    r"""Project the Euclidean gradient onto the sphere's tangent space at `x`.

    Requires $‖x‖=1$ along the final axis.
    """
    return grad_euclidean - x * vecdot(grad_euclidean, x, dim=-1).unsqueeze(-1)


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
    status: OptimizerStatus
    fun: Tensor | float
    jac: Tensor

    msg: str = ""
    nit: int | None = None
    max_cv: Tensor | float | None = None
    loss_hist: list[float] = field(default_factory=list)
    grad_hist: list[float] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self):
        return self.status is OptimizerStatus.SUCCESS


def _scaled_norm(x: Tensor, /, *, axis: None | int | tuple[int, ...] = None) -> Tensor:
    r"""Compute the root mean square $√((∑ᵢ xᵢ²)/n)$ over the selected axes.

    Here $n$ counts the reduced elements. With `axis=None`, reduce all axes.
    """
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
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
    seed: Optional[int] = None,
) -> OptimizerResult:
    r"""Optimize $n$ points on the unit sphere $Sᵈ⁻¹⊂ℝᵈ$ for separation.

    Uses `separation_loss` to aggregate off-diagonal inner products, starting
    from `wide_angle_sphere_init`. The gradients are projected onto the tangent space
    and then the points are updated along sphere geodesics.
    The loss is inspired by the Thomson problem but differs from its inverse-distance energy.

    $$  𝓛 = \softmaxᵦ(XXᵀ - 2𝕀ₙ) ≈ \max_{i≠j} ⟨xᵢ∣xⱼ⟩  $$

    Special cases use direct constructions: random signs for `dim=1`,
    equally spaced circle points for `dim=2`, a single random unit vector
    for `num=1`, and antipodal points for `num=2` in higher dimensions.
    For `dim=1`, repeated points are possible. For `num=1`, the separation
    loss is $-∞$ because there are no off-diagonal pairs.

    Args:
        num: Number of points $n≥1$.
        dim: Ambient dimension $d≥1$; points lie on $Sᵈ⁻¹⊂ℝᵈ$.
        beta: Separation-loss parameter $β>1$. See `separation_loss` for its scaling.
        maxiter: Maximum number of optimization iterations.
        atol: Absolute tolerance for convergence and unit-norm constraints.
        rtol: Relative tolerance for convergence and unit-norm constraints.
        patience: Number of iterations to look back when checking convergence.
        dtype: Data type of the returned points and optimization state.
        device: Device of the returned points and optimization state.
        seed: Seed passed to `wide_angle_sphere_init` in the iterative case.

    Returns:
        Optimization result containing the points, loss, gradient, status,
        and iteration history. Direct constructions return without iteration
        history; `seed` is only applied in the iterative case.
    """
    loss_fn = partial(separation_loss, beta=beta)
    grad_fn = torch.func.grad(loss_fn)
    grad_and_value_fn = torch.func.grad_and_value(loss_fn)
    options = {"beta": beta, "atol": atol, "rtol": rtol, "maxiter": maxiter}

    # special case dim≤2 and num≤2
    if dim == 1:
        # sample from {-1, +1} with equal probability, which is optimal for d=1
        points = 2.0 * torch.randint(0, 2, (num, 1), device=device) - 1.0
        points = points.to(dtype=dtype)
        grad, loss = grad_and_value_fn(points)
        return OptimizerResult(
            points, fun=loss, jac=grad, status=OptimizerStatus.SUCCESS, options=options
        )

    if dim == 2:
        # for d=2, we can just put the points uniformly on the circle, which is optimal
        angles = torch.linspace(
            0,
            2 * math.pi,
            num + 1,
            dtype=dtype,
            device=device,
        )[:-1]
        random_phase = 2 * math.pi * torch.rand((), dtype=dtype, device=device)
        angles = angles + random_phase
        points = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        grad, loss = grad_and_value_fn(points)
        return OptimizerResult(
            points, fun=loss, jac=grad, status=OptimizerStatus.SUCCESS, options=options
        )

    if num == 1:
        # sample a random point on the sphere, which is optimal
        points = torch.randn(1, dim, dtype=dtype, device=device)
        points = points / vector_norm(points, dim=-1, keepdim=True)
        grad, loss = grad_and_value_fn(points)
        return OptimizerResult(
            points, fun=loss, jac=grad, status=OptimizerStatus.SUCCESS, options=options
        )

    if num == 2:
        # sample a random point and the antipodal point
        points = torch.randn(1, dim, dtype=dtype, device=device)
        points = points / vector_norm(points, dim=-1, keepdim=True)
        points = torch.cat([points, -points], dim=0)
        grad, loss = grad_and_value_fn(points)
        return OptimizerResult(
            points, fun=loss, jac=grad, status=OptimizerStatus.SUCCESS, options=options
        )

    def _body_fn(z: Tensor, /) -> Tensor:
        # Negative Euclidean gradient by autodiff
        g_euclidean = -grad_fn(z)
        # Convert to Riemannian gradient on tangent space of the sphere
        g = sphere_gradient(z, grad_euclidean=g_euclidean)

        # normalize the gradient to get a unit direction on the sphere
        g_norm = vector_norm(g, dim=-1, keepdim=True)
        g = g / (g_norm + 1e-8)

        # apply retraction (exponential map) to update the points on the sphere
        # For the sphere, the geodesic from x in the direction of g is:
        # γ(t) = cos(‖g‖t)x + sin(‖g‖t)g/‖g‖
        # assuming g is normalized, this is
        # γ(t) = cos(t)x + sin(t)g
        # line search to find the optimal step size along the geodesic
        lowers = torch.zeros(z.shape[:-1], dtype=z.dtype, device=z.device)
        uppers = torch.full(z.shape[:-1], math.pi / 4, dtype=z.dtype, device=z.device)
        t = safe_bisection_line_search(
            lambda s: loss_fn(sphere_geodesic(s, z, g)),
            lowers,
            uppers,
            maxiter=10,
        )
        z = sphere_geodesic(t, start=z, direction=g)

        # normalize the points to ensure they are on the sphere
        return z / vector_norm(z, dim=-1, keepdim=True)

    # general case.
    x = wide_angle_sphere_init(num, dim, dtype=dtype, device=device, seed=seed)
    grad, loss = grad_and_value_fn(x)
    grad = sphere_gradient(x, grad_euclidean=grad)
    grad_norm = _scaled_norm(grad, axis=-1).sum()
    loss_value = float(loss)
    grad_norm_value = float(grad_norm)

    # initialize the history lists with the initial loss and gradient norm
    loss_hist: list[float] = [float(loss)]
    grad_hist: list[float] = [float(grad_norm)]

    it = 0
    for it in range(maxiter):
        x = _body_fn(x)

        # track iteration progress
        grad, loss = grad_and_value_fn(x)
        grad = sphere_gradient(x, grad_euclidean=grad)
        grad_norm = _scaled_norm(grad, axis=-1).sum()
        loss_value = float(loss)
        grad_norm_value = float(grad_norm)
        loss_hist.append(loss_value)
        grad_hist.append(grad_norm_value)

        if not (loss.isfinite() & grad_norm.isfinite()):
            msg = (
                f"Warning: Non-finite value encountered at iteration {it}."
                f" loss={loss_value:.6f}, grad_norm={grad_norm_value:.6f}"
            )
            status = OptimizerStatus.NON_FINITE_VALUES
            break

        if not torch.allclose(
            vector_norm(x, dim=-1),
            torch.ones(x.shape[0], dtype=x.dtype, device=x.device),
            atol=atol,
            rtol=rtol,
        ):
            max_deviation = torch.abs(vector_norm(x, dim=-1) - 1).max().item()
            msg = (
                f"Warning: Constraint violation at iteration {it}."
                f" Points not on the sphere: {max_deviation=:.6f}"
            )
            status = OptimizerStatus.CONSTRAINT_VIOLATION
            break

        m = -min(patience + 1, it + 2)  # look back at most p iterations
        if (grad_hist[m] - grad_norm_value) < rtol * grad_hist[m] + atol:
            msg = (
                f"Converged after {it} iterations."
                f" Final loss: {loss_value:.6f}"
                f" Final (per particle) grad norm: {grad_norm_value:.6f}"
            )
            status = OptimizerStatus.SUCCESS
            break

        if loss_value >= loss_hist[-1] + atol:
            raise RuntimeError(
                f"{it=} Loss did not decrease: {loss_value:.6f} >= {loss_hist[-1]:.6f}"
            )
    else:
        status = OptimizerStatus.NO_CONVERGENCE
        msg = (
            f"Warning: Did not converge after {maxiter} iterations."
            f" Final loss: {loss_value:.6f}"
            f" Final (per particle) grad norm: {grad_norm_value:.6f}"
        )

    max_constraint_violation = (vector_norm(x, dim=-1) - 1).abs().max()
    result = OptimizerResult(
        x=x,
        fun=loss,
        jac=grad,
        status=status,
        msg=msg,
        nit=it,
        loss_hist=loss_hist,
        grad_hist=grad_hist,
        max_cv=max_constraint_violation,
        options=options,
    )

    return result
