r"""Test testing."""

import torch

__all__ = [
    # Functions
    "as_seed",
    "as_torch_generator",
    "camel2snake",
    "snake2camel",
]


def as_seed(rng: int | torch.Generator, /) -> int:
    r"""Return an integer seed for ``rng``."""
    if isinstance(rng, int):
        return rng
    if isinstance(rng, torch.Generator):
        return int(rng.initial_seed())
    raise TypeError(f"Expected int or torch.Generator, got {type(rng)!r}.")


def as_torch_generator(
    rng: int | torch.Generator, /, *, device: str | torch.device = "cpu"
) -> torch.Generator:
    r"""Return a torch generator for ``rng`` on ``device`` when seeded by an int."""
    if isinstance(rng, torch.Generator):
        return rng
    if isinstance(rng, int):
        generator = torch.Generator(device=device)
        generator.manual_seed(rng)
        return generator
    raise TypeError(f"Expected int or torch.Generator, got {type(rng)!r}.")


def camel2snake(string: str) -> str:
    r"""Convert camel case to snake case."""
    return "".join(["_" + c.lower() if c.isupper() else c for c in string]).lstrip("_")


def snake2camel(string: str) -> str:
    r"""Convert snake case to camel case."""
    return "".join([c[0].upper() + c[1:] for c in string.split("_")])
