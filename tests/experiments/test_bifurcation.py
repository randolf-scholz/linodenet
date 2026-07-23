r"""Generate and visualize Brownian trajectories with a random drift bifurcation."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from torch import Tensor

from tests.testing import PROJECT

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


@dataclass(frozen=True, slots=True)
class BifurcationSampler:
    r"""Sample Brownian trajectories with a randomly signed drift after $t⁎$.

    The sampled values satisfy $Y(t) = σW(t) + Cβ(t - t⁎)₊$, where $W$ is a
    standard Brownian motion and $C ∈ {-1, 1}$ is sampled independently for
    every trajectory.
    """

    t_star: float
    beta: float
    sigma: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.t_star <= 1.0:
            msg = f"Expected t_star in [0, 1], got {self.t_star}."
            raise ValueError(msg)
        if self.beta < 0.0:
            msg = f"Expected beta to be non-negative, got {self.beta}."
            raise ValueError(msg)
        if self.sigma < 0.0:
            msg = f"Expected sigma to be non-negative, got {self.sigma}."
            raise ValueError(msg)

    def sample(
        self, num_trajectories: int, num_steps: int, /
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Return sorted times, trajectory values, and drift signs.

        Args:
            num_trajectories: Number of independent Brownian trajectories.
            num_steps: Number of randomly sampled time steps per trajectory.

        Returns:
            A triple $(T, Y, C)$ with shapes $(N, S)$, $(N, S)$, and $(N,)$.
        """
        if num_trajectories < 1:
            msg = f"Expected at least one trajectory, got {num_trajectories}."
            raise ValueError(msg)
        if num_steps < 1:
            msg = f"Expected at least one time step, got {num_steps}."
            raise ValueError(msg)

        times = torch.rand(num_trajectories, num_steps).sort(dim=-1).values
        time_deltas = torch.diff(times, dim=-1, prepend=torch.zeros_like(times[:, :1]))
        brownian_motion = torch.cumsum(
            self.sigma * torch.randn_like(times) * time_deltas.sqrt(), dim=-1
        )
        coin_result = (
            torch.randint(0, 2, (num_trajectories,), dtype=torch.int64) * 2 - 1
        )
        drift = coin_result[:, None] * self.beta * (times - self.t_star).clamp_min(0)
        return times, brownian_motion + drift, coin_result


def test_samples() -> None:
    r"""Visualize Brownian trajectories after their randomly signed bifurcation."""
    torch.manual_seed(0)
    sampler = BifurcationSampler(t_star=0.3, beta=2.0, sigma=0.2)
    times, values, coin_results = sampler.sample(100, 100)

    assert times.shape == values.shape == (100, 100)
    assert coin_results.shape == (100,)
    assert torch.all(times[:, 1:] >= times[:, :-1])

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for time, value, coin_result in zip(times, values, coin_results, strict=True):
        color = "tab:blue" if coin_result.item() < 0 else "tab:orange"
        ax.plot(time, value, color=color, alpha=0.35, linewidth=0.8)

    ax.axvline(sampler.t_star, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="time", ylabel="value", xlim=(0, 1), title="Brownian-motion bifurcation"
    )
    ax.legend(
        handles=[
            Line2D([], [], color="tab:blue", label="negative drift"),
            Line2D([], [], color="tab:orange", label="positive drift"),
        ]
    )
    fig.savefig(RESULT_DIR / "bifurcation_samples.png", dpi=200)
    plt.close(fig)
