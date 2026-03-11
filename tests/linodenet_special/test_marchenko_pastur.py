import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from linodenet_special.marchenko_pastur import MarchenkoPastur
from tests.utils.project import PROJECT


@pytest.mark.parametrize(
    ("m", "n"),
    [
        (512, 1024),
        (512, 768),
        (512, 512),
        (768, 512),
        (1024, 512),
    ],
    # ids=["square", "m>n", "n>m", "gamma=0.5", "gamma=2.0"],
)
def test_marchenko_pastur(m: int, n: int) -> None:
    mpl.use("Agg")
    torch.manual_seed(0)

    xmin = -1.0
    xmax = 6.0
    gamma = m / n
    sigma2 = 1.0

    x = torch.randn(m, n, dtype=torch.float64) / math.sqrt(n)
    s = torch.linalg.svdvals(x)
    s2 = (s**2).cpu().numpy()

    dist = MarchenkoPastur(gamma=gamma, sigma2=sigma2, validate_args=False)
    bins = 64
    hist, edges = np.histogram(s2, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdf_x = np.linspace(xmin, xmax, 512)
    pdf = torch.exp(dist.log_prob(torch.from_numpy(pdf_x))).numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(centers, hist, width=edges[1] - edges[0], alpha=0.5, label="Empirical")
    ax.plot(pdf_x, pdf, lw=2, color="C1", label="MP pdf")
    ax.set_title(f"Marchenko-Pastur (gamma={gamma:.3f})")
    ax.set_xlabel("Singular value squared")
    ax.set_ylabel("Density")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 2)
    ax.legend()

    result_dir = PROJECT.RESULTS_DIR[__file__]
    result_dir.mkdir(exist_ok=True)
    out = result_dir / f"marchenko_pastur_{m}x{n}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert out.exists()
