# SEE: https://github.com/pytorch/pytorch/issues/167645

import gc
import tempfile
import time

import torch
from torch import nn
from torch.nn.utils import parametrizations, parametrize


class Foo(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = n = input_size
        self.hidden_size = m = hidden_size

        # submodules with expensive parametrizations
        self.propagator = parametrizations.orthogonal(nn.Linear(m, m))
        self.decoder = parametrizations.spectral_norm(
            nn.Linear(m, n), n_power_iterations=10
        )
        self.encoder = parametrizations.spectral_norm(
            nn.Linear(n, m), n_power_iterations=10
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""(*Bs, L, D) - > (*Bs, L, D)."""
        x = torch.zeros(self.hidden_size, device=inputs.device)

        for y in inputs.unbind(-2):
            x = self.propagator(x)
            yhat = self.decoder(x)
            x = x + self.encoder(yhat - y)

        return x


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
    device = torch.device("cuda")
    batch_size = 32
    sequence_length = 512
    hidden_size = 128
    input_size = 64
    model = Foo(input_size, hidden_size).to(device)
    inputs = torch.randn(batch_size, sequence_length, input_size, device=device)

    # warmup
    model(inputs)

    with timer("Uncompiled without caching: {:.3f} s"):
        model(inputs)

    with (
        parametrize.cached(),
        timer("Uncompiled with caching: {:.3f} s"),
    ):
        model(inputs)

    with timer("Compiled (torch.compile): {:.3f} s"):
        compiled = torch.compile(model, fullgraph=True)

    # warmup
    compiled(inputs)

    with timer("Compiled (torch.compile) without caching: {:.3f} s"):
        compiled(inputs)

    with (
        parametrize.cached(),
        timer("Compiled (torch.compile) with caching: {:.3f} s"),
    ):
        compiled(inputs)

    with timer("Compiling (torch.exoport): {:.3f} s"):
        exported = torch.export.export(
            model,
            args=(inputs,),
            dynamic_shapes={
                "inputs": {
                    -3: torch.export.Dim("batch"),
                    -2: torch.export.Dim("sequence_length"),
                }
            },
            strict=True,
        )

    with (
        tempfile.TemporaryFile() as tmp,
        timer("Compiled export+import time: {:.3f} s"),
    ):
        torch.export.save(exported, tmp)
        tmp.seek(0)
        deserialized_model = torch.export.load(tmp).module()

    # warmup
    deserialized_model(inputs)

    with timer("Compiled (export) without caching: {:.3f} s"):
        deserialized_model(inputs)

    with (
        parametrize.cached(),
        timer("Compiled (export) with caching: {:.3f} s"),
    ):
        deserialized_model(inputs)


if __name__ == "__main__":
    test()
