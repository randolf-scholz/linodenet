__all__ = ["DEVICES", "DTYPES", "SEEDS"]

import torch

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
DTYPES = [torch.float32, torch.float64]
SEEDS = [1000, 1001, 1002, 1003, 1004]
