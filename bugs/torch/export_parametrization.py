import gc
import tempfile
import time

import torch
from torch.nn.utils import parametrizations


class timer:
    def __init__(self, msg: str = "Elapsed: {:.6f} s"):
        self.msg = msg

    def __enter__(self):
        torch.cuda.synchronize()
        gc.disable()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0
        gc.enable()
        print(self.msg.format(self.elapsed))


def test():
    model = torch.nn.Linear(10, 10)
    args = torch.randn(1, 10)

    with timer():
        torch.export.export(
            model, args=(args,), dynamic_shapes={0: torch.export.Dim("batch")}
        )

    parametrized = parametrizations.spectral_norm(model)

    with timer():
        torch.export.export(
            parametrized, args=(args,), dynamic_shapes={0: torch.export.Dim("batch")}
        )


if __name__ == "__main__":
    test()
