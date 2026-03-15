import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from scipy.stats import ortho_group

from linodenet_special.marchenko_pastur import MarchenkoPastur
from tests.utils.project import PROJECT

from .fixtures import DTYPES, SEED


@pytest.mark.parametrize(
    "shape",
    [
        (512, 1024),
        (512, 768),
        (512, 512),
        (768, 512),
        (1024, 512),
    ],
    ids=lambda x: f"{x[0]}x{x[1]}",
)
def test_plot_marchenko_pastur(shape: tuple[int, int]) -> None:
    mpl.use("Agg")
    torch.manual_seed(0)

    X_MIN, X_MAX = -1.0, 6.0
    Y_MIN, Y_MAX = 0.0, 1.5
    m, n = shape
    gamma = m / n
    sigma2 = 1.0

    x = torch.randn(m, n, dtype=torch.float64) / math.sqrt(n)
    s = torch.linalg.svdvals(x)
    s2 = (s**2).cpu().numpy()

    dist = MarchenkoPastur(gamma=gamma, sigma2=sigma2, validate_args=False)
    x = torch.linspace(X_MIN, X_MAX, 512)
    pdf = torch.exp(dist.log_prob(x))

    if gamma > 1:
        # need the conditional distribution x>0, since singular values are by definition positive
        pdf = pdf / (1.0 - dist.point_mass)
        pdf = torch.where(x > 0, pdf, 0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(s2, density=True, alpha=0.5, bins=64, label="Empirical")
    ax.plot(x, pdf, lw=2, color="C1", label="MP pdf")
    ax.set_title(f"Marchenko-Pastur(γ={gamma:.3f} | x>0)")
    ax.set_xlabel(f"Singular value squared ({m}x{n}, entries ~ N(0, 1/√({n}))")
    ax.set_ylabel("Density")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.legend()

    result_dir = PROJECT.RESULTS_DIR[__file__]
    result_dir.mkdir(exist_ok=True)
    out = result_dir / f"marchenko_pastur_{m}x{n}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert out.exists()


@pytest.mark.parametrize(
    ("gamma", "sigma2"),
    [(2.0, 1.0), (0.5, 1.0), (2.0, 3.0)],
    ids=str,
)
def test_marchenko_pastur_sample_positive(gamma: float, sigma2: float) -> None:
    torch.manual_seed(SEED)
    dist = MarchenkoPastur(gamma=gamma, sigma2=sigma2, validate_args=False)
    samples = dist.sample_positive((4096,))

    assert torch.all(samples > 0)
    assert torch.all(samples >= dist.lower_bound)
    assert torch.all(samples <= dist.upper_bound)


def test_marchenko_pastur_support_includes_zero_atom() -> None:
    dist = MarchenkoPastur(gamma=2.0, sigma2=1.0, validate_args=True)

    assert dist.support.check(torch.tensor(0.0))
    assert dist.support.check(dist.lower_bound)
    assert not dist.support.check(torch.tensor(-1.0))
    assert torch.isfinite(dist.log_prob(torch.tensor(0.0)))


@pytest.mark.flaky(returns=3)
@pytest.mark.parametrize("batch", [512], ids="batch={}".format)
@pytest.mark.parametrize("seed", [SEED], ids="seed={}".format)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(
    "shape", [(64, 128), (128, 128), (128, 64)], ids=lambda x: f"{x[0]}x{x[1]}"
)
def test_matrix_construction_from_marchenko_pastur(
    batch: int, shape: tuple[int, int], dtype, seed: int
) -> None:
    r"""Tests that matrix construction A = USVᵀ entries are approximately N(0, 1/n) distributed."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m, n = shape
    k = min(m, n)
    gamma = m / n

    # only the first k vectors
    U_numpy = ortho_group(m).rvs(size=batch, random_state=rng)[..., :k]
    V_numpy = ortho_group(n).rvs(size=batch, random_state=rng)[..., :k]
    U = torch.from_numpy(U_numpy).to(dtype=dtype)
    V = torch.from_numpy(V_numpy).to(dtype=dtype)
    dist = MarchenkoPastur(gamma=gamma, sigma2=1.0, validate_args=False)
    S = dist.sample_positive((batch, k)).to(dtype=dtype).sqrt()
    A = torch.einsum("...mk, ...k, ...nk ->  ...mn", U, S, V)

    mean = A.mean(dim=0)
    std = A.std(dim=0, unbiased=False)
    expected_mean = 0.0
    expected_std = 1.0 / math.sqrt(n)

    num_elements = m * n
    # Uses CLT scaling and a union-bound style max correction for m*n elements.
    # Mean SE: sqrt(1/(n*batch)); std SE: (1/sqrt(n)) / sqrt(2*(batch-1)).
    # Inflate by sqrt(2 log(m*n)) for max over elements, and a safety factor of 5.
    max_factor = math.sqrt(2.0 * math.log(max(num_elements, 2)))
    mean_tol = 5.0 * max_factor * math.sqrt(1.0 / (n * batch))
    std_tol = 5.0 * max_factor * expected_std / math.sqrt(2.0 * max(batch - 1, 1))
    print(f"{batch=} {m=} {n=} {mean_tol=:.4e} {std_tol=:.4e}")
    mean_diff = (mean - expected_mean).abs()
    std_diff = (std - expected_std).abs()

    assert torch.all(mean_diff <= mean_tol), (
        "Mean out of range: "
        f"max_mean={mean.max().item():.4e}, min_mean={mean.min().item():.4e}, "
        f"max_diff={mean_diff.max().item():.4e}, tol={mean_tol:.4e}"
    )
    assert torch.all(std_diff <= std_tol), (
        "Std out of range: "
        f"max_std={std.max().item():.4e}, min_std={std.min().item():.4e}, "
        f"max_diff={std_diff.max().item():.4e}, tol={std_tol:.4e}"
    )


@pytest.mark.parametrize(
    ("m", "n"),
    [(256, 128), (128, 256), (256, 256)],
)
@pytest.mark.parametrize("dtype", [torch.float64], ids=str)
def test_marchenko_pastur_xxt_eigenvalues(m: int, n: int, dtype) -> None:
    r"""Compare MP law to eigenvalues of X Xᵀ (including the zero point mass)."""
    torch.manual_seed(SEED)
    gamma = m / n

    x = torch.randn(m, n, dtype=dtype) / math.sqrt(n)
    s = torch.linalg.svdvals(x)
    s2 = s**2
    if m > n:
        s2 = torch.cat([s2, torch.zeros(m - n, dtype=dtype)])

    dist = MarchenkoPastur(gamma=gamma, sigma2=1.0, validate_args=False)
    point_mass = dist.point_mass.item()
    eps = 1e-12
    empirical_mass = (s2 <= eps).to(dtype=dtype).mean().item()
    mass_tol = 0.05 if point_mass > 0 else 0.02
    assert abs(empirical_mass - point_mass) <= mass_tol, (
        "Point mass mismatch: "
        f"empirical={empirical_mass:.4f}, expected={point_mass:.4f}, tol={mass_tol:.4f}"
    )

    empirical_mean = s2.mean()
    empirical_var = s2.var(unbiased=False)
    expected_mean = dist.mean
    expected_var = dist.variance
    mean_tol = max(5e-3, 5.0 * math.sqrt(expected_var.item() / m))
    var_tol = 0.25 * expected_var.item() + 5e-4

    assert (empirical_mean - expected_mean).abs().item() <= mean_tol, (
        "Mean out of range: "
        f"mean={empirical_mean.item():.4e}, expected={expected_mean.item():.4e}, "
        f"tol={mean_tol:.4e}"
    )
    assert (empirical_var - expected_var).abs().item() <= var_tol, (
        "Variance out of range: "
        f"var={empirical_var.item():.4e}, expected={expected_var.item():.4e}, "
        f"tol={var_tol:.4e}"
    )
