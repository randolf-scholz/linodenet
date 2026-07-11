__all__ = [
    "DEVICES",
    "PREFER_GPU",
    "DTYPES",
    "SEEDS_5",
    "SEEDS_3",
    "SEED",
    "SEEDS_10",
]

from typing import Final

import torch

PREFER_GPU: Final[list[str]] = ["cuda"] if torch.cuda.is_available() else ["cpu"]
DEVICES: Final[list[str]] = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
DTYPES: Final[list[torch.dtype]] = [torch.float32, torch.float64]
SEEDS_5: Final[list[int]] = [1000, 1001, 1002, 1003, 1004]
SEEDS_3: Final[list[int]] = [1000, 1001, 1002]
SEED: Final[int] = 0
SEEDS_10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
