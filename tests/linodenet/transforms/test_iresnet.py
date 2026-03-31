import logging

import matplotlib.pyplot as plt
import pytest
import torch
from torch import Tensor
from torch.nn.functional import mse_loss
from torchinfo import summary

from linodenet.mappings.transforms import IResNet, ResidualContraction
from linodenet.nn.parametrize import update_parametrizations
from tests.testing import DEVICES, DTYPES, PROJECT, visualize_distribution

from .test_transform import TestTransform

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


def make_model(
    input_size: int,
    /,
    *,
    num_blocks: int,
    layers_per_block: int,
    latent_size: int,
    use_rezero: bool,
    maxiter: int,
    atol: float,
    rtol: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> IResNet:
    model = IResNet(
        input_size,
        num_blocks=num_blocks,
        layers_per_block=layers_per_block,
        latent_size=latent_size,
        activation="ReLU",
        use_rezero=use_rezero,
        maxiter=maxiter,
        atol=atol,
        rtol=rtol,
    )
    model = model.to(device=device, dtype=dtype)
    update_parametrizations(model)
    return model


def train_model(
    model: IResNet,
    /,
    *,
    x: Tensor,
    target: Tensor,
    steps: int,
    learning_rate: float,
) -> None:
    if model.use_rezero:
        parameters = [
            block.scalar
            for block in model
            if isinstance(block, ResidualContraction) and block.scalar is not None
        ]
    else:
        parameters = list(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = mse_loss(model.encode(x), target)
        loss.backward()
        optimizer.step()
        update_parametrizations(model)


def compute_inversion_errors(
    model: IResNet, x: Tensor, y: Tensor, /
) -> dict[str, Tensor]:
    fx = model.encode(x)
    xhat = model.decode(fx)
    ify = model.decode(y)
    yhat = model.encode(ify)
    return {
        "forward_inverse_error": torch.linalg.vector_norm(x - xhat, dim=-1),
        "inverse_forward_error": torch.linalg.vector_norm(y - yhat, dim=-1),
        "forward_difference": torch.linalg.vector_norm(x - fx, dim=-1),
        "inverse_difference": torch.linalg.vector_norm(y - ify, dim=-1),
    }


@pytest.mark.parametrize("use_rezero", [False, True], ids=["plain", "rezero"])
def test_instantiation(use_rezero: bool) -> None:
    input_size = 8
    num_blocks = 3
    layers_per_block = 3
    latent_size = 16
    model = make_model(
        input_size,
        num_blocks=num_blocks,
        layers_per_block=layers_per_block,
        latent_size=latent_size,
        use_rezero=use_rezero,
        maxiter=128,
        atol=1e-6,
        rtol=1e-6,
    )

    assert model.input_size == input_size
    assert model.num_blocks == num_blocks
    assert model.layers_per_block == layers_per_block
    assert model.latent_size == latent_size
    assert model.use_rezero is use_rezero
    assert len(model) == num_blocks
    assert all(isinstance(block, ResidualContraction) for block in model)

    stats = summary(
        model,
        input_size=(4, input_size),
        dtypes=[torch.float32],
        verbose=0,
    )
    print(stats)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_rezero", [False, True], ids=["plain", "rezero"])
class TestIResNetInvertibility(TestTransform):
    INPUT_SIZE = 8
    BATCH_SIZE = 64
    NUM_BLOCKS = 3
    LAYERS_PER_BLOCK = 3
    LATENT_SIZE = 16
    TRAIN_STEPS = 20
    LEARNING_RATE = 1.0
    QUANTILES = torch.tensor([0.5, 0.68, 0.95, 0.997])
    ERROR_TARGETS = {
        torch.float32: torch.tensor([1e-6, 2e-6, 1e-5, 1e-4]),
        torch.float64: torch.tensor([1e-9, 1e-8, 1e-7, 5e-7]),
    }
    FLOW_TOL = {
        torch.float32: (1e-6, 1e-6),
        torch.float64: (1e-8, 1e-8),
    }

    def evaluate_invertibility(
        self,
        model: IResNet,
        /,
        *,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        max_error = float(self.ERROR_TARGETS[dtype][-1])
        x = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device, dtype=dtype)
        y = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device, dtype=dtype)
        self.assert_invertible(
            model,
            x,
            y,
            atol=max_error,
            rtol=max_error,
            logdet_atol=max_error,
            logdet_rtol=max_error,
        )
        metrics = compute_inversion_errors(model, x, y)
        quantiles = self.QUANTILES.to(device=device, dtype=dtype)

        forward_inverse_quantiles = torch.quantile(
            metrics["forward_inverse_error"], quantiles
        ).cpu()
        inverse_forward_quantiles = torch.quantile(
            metrics["inverse_forward_error"], quantiles
        ).cpu()

        assert torch.all(forward_inverse_quantiles <= self.ERROR_TARGETS[dtype])
        assert torch.all(inverse_forward_quantiles <= self.ERROR_TARGETS[dtype])

    def test_invertible_at_initialization(
        self,
        dtype: torch.dtype,
        device: str,
        use_rezero: bool,
    ) -> None:
        atol, rtol = self.FLOW_TOL[dtype]
        model = make_model(
            self.INPUT_SIZE,
            num_blocks=self.NUM_BLOCKS,
            layers_per_block=self.LAYERS_PER_BLOCK,
            latent_size=self.LATENT_SIZE,
            use_rezero=use_rezero,
            maxiter=256,
            atol=atol,
            rtol=rtol,
            device=device,
            dtype=dtype,
        )
        self.evaluate_invertibility(model, dtype=dtype, device=device)

    def test_invertible_after_training(
        self,
        dtype: torch.dtype,
        device: str,
        use_rezero: bool,
    ) -> None:
        torch.manual_seed(0)
        atol, rtol = self.FLOW_TOL[dtype]
        model = make_model(
            self.INPUT_SIZE,
            num_blocks=self.NUM_BLOCKS,
            layers_per_block=self.LAYERS_PER_BLOCK,
            latent_size=self.LATENT_SIZE,
            use_rezero=use_rezero,
            maxiter=256,
            atol=atol,
            rtol=rtol,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            train_x = torch.randn(
                self.BATCH_SIZE, self.INPUT_SIZE, device=device, dtype=dtype
            )
            train_target = torch.randn(
                self.BATCH_SIZE, self.INPUT_SIZE, device=device, dtype=dtype
            )
            initial_loss = mse_loss(model.encode(train_x), train_target)

        train_model(
            model,
            x=train_x,
            target=train_target,
            steps=self.TRAIN_STEPS,
            learning_rate=self.LEARNING_RATE,
        )
        with torch.no_grad():
            final_loss = mse_loss(model.encode(train_x), train_target)
            movement = torch.linalg.vector_norm(model.encode(train_x) - train_x, dim=-1)

        assert final_loss < initial_loss
        assert torch.quantile(movement, 0.5) > 1e-3
        self.evaluate_invertibility(model, dtype=dtype, device=device)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_rezero", [False], ids=["plain"])
def test_plot_errors(
    use_rezero: bool,
    device: str,
    dtype: torch.dtype,
) -> None:
    logger = logging.getLogger(f"{__name__}/{IResNet.__name__}")
    logger.info("Testing plot generation for use_rezero=%s", use_rezero)

    input_size = 8
    num_blocks = 3
    layers_per_block = 3
    latent_size = 16
    batch_size = 2_048
    maxiter = 256
    atol = 1e-6 if dtype is torch.float32 else 1e-8
    rtol = atol
    extra_stats = {
        "Samples": f"{batch_size}",
        "Dim": f"{input_size}",
        "Blocks": f"{num_blocks}",
        "Layers": f"{layers_per_block}",
        "Latent": f"{latent_size}",
        "ReZero": str(use_rezero),
        "Device": device,
        "DType": str(dtype),
        "maxiter": f"{maxiter}",
    }

    model = make_model(
        input_size,
        num_blocks=num_blocks,
        layers_per_block=layers_per_block,
        latent_size=latent_size,
        use_rezero=use_rezero,
        maxiter=maxiter,
        atol=atol,
        rtol=rtol,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        x = torch.randn(batch_size, input_size, device=device, dtype=dtype)
        y = torch.randn(batch_size, input_size, device=device, dtype=dtype)
        metrics = compute_inversion_errors(model, x, y)

    fig, ax = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=(12, 12),
        tight_layout=True,
        sharex="row",
        sharey="row",
        squeeze=False,
    )

    visualize_distribution(
        metrics["forward_inverse_error"], ax=ax[0, 0], extra_stats=extra_stats
    )
    visualize_distribution(
        metrics["inverse_forward_error"], ax=ax[0, 1], extra_stats=extra_stats
    )
    visualize_distribution(
        metrics["forward_difference"], ax=ax[1, 0], extra_stats=extra_stats
    )
    visualize_distribution(
        metrics["inverse_difference"], ax=ax[1, 1], extra_stats=extra_stats
    )

    ax[0, 0].set_xlabel(
        r"$r_\mathrm{left}(x) = \|x - \phi^{-1}(\phi(x))\|$ where $x_i \sim \mathcal{N}(0,1)$"
    )
    ax[0, 0].set_ylabel(r"density $p(r_\mathrm{left} \mid x)$")
    ax[0, 1].set_xlabel(
        r"$r_\mathrm{right}(y) = \|y - \phi(\phi^{-1}(y))\|$ where $y_j \sim \mathcal{N}(0,1)$"
    )
    ax[0, 1].set_ylabel(r"density $p(r_\mathrm{right} \mid y)$")
    ax[1, 0].set_xlabel(
        r"$d_\mathrm{left}(x) = \|x - \phi(x)\|$ where $x_i \sim \mathcal{N}(0,1)$"
    )
    ax[1, 0].set_ylabel(r"density $p(d_\mathrm{left} \mid x)$")
    ax[1, 1].set_xlabel(
        r"$d_\mathrm{right}(y) = \|y - \phi^{-1}(y)\|$ where $y_j \sim \mathcal{N}(0,1)$"
    )
    ax[1, 1].set_ylabel(r"density $p(d_\mathrm{right} \mid y)$")
    fig.suptitle(
        f"IResNet -- Inversion Property (use_rezero={use_rezero}, device={device}, dtype={dtype})",
        fontsize=16,
    )
    dtype_name = str(dtype).rsplit(".", maxsplit=1)[-1]
    stem = f"iresnet_rezero={use_rezero}_{device}_{dtype_name}"
    fig.savefig(RESULT_DIR / f"{stem}.pdf")
    fig.savefig(RESULT_DIR / f"{stem}.svg")
    fig.savefig(RESULT_DIR / f"{stem}.png", dpi=300)
