r"""Test if model initializations, forward and backward passes."""

import logging
from itertools import product

import pytest
import torch
from torch import Tensor, nn

from linodenet.flows import LinearFlow
from linodenet.forecasting import LinODEnet
from linodenet.mappings.transforms import iResNet, iResNetBlock
from linodenet.nn import LinearContraction
from linodenet.testing import assert_model_ok, check_initialization
from tests.testing import PROJECT

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]

OUTER_BATCH = 3
INNER_BATCH = 5
LEN = 9  # sequence LENgth
DIM = 7  # input DIMension
OUT = 6  # OUTput dimension
LAT = 8  # LATent dimension

DTYPE = torch.float32
DEVICES = [torch.device("cpu")]

# if torch.cuda.is_available():
#     DEVICES.append(torch.device("cuda"))

BATCH_SIZES = [(), (INNER_BATCH,), (OUTER_BATCH, INNER_BATCH)]

MODELS: dict[type[nn.Module], dict] = {
    LinearContraction: {
        "init_args": (DIM, OUT),
        "init_kwargs": {},
        "input_shapes": [(LEN, DIM)],  # X
        "output_shapes": [(LEN, OUT)],
    },
    iResNetBlock: {
        "init_args": (DIM,),
        "init_kwargs": {},
        "input_shapes": [(LEN, DIM)],  # X
        "output_shapes": [(LEN, DIM)],
    },
    iResNet: {
        "init_args": (DIM,),
        "init_kwargs": {},
        "input_shapes": [(LEN, DIM)],  # X
        "output_shapes": [(LEN, DIM)],
    },
    LinearFlow: {
        "init_args": (DIM,),
        "init_kwargs": {},
        "input_shapes": [(), (DIM,)],  # Δt, x0
        "output_shapes": [(DIM,)],
    },
    LinODEnet: {
        "init_args": (DIM, LAT),
        "init_kwargs": {},
        "input_shapes": [(LEN,), (LEN, DIM)],  # T, X
        "output_shapes": [(LEN, DIM)],
    },
}


def _make_tensors(
    shapes: list[tuple[int, ...]],
    *,
    batch_sizes: tuple[int, ...] = (),
    dtype: torch.dtype = torch.float32,
    device: torch.device = DEVICES[0],
) -> tuple[Tensor, ...]:
    r"""Random tensors of required shape with potentially multiple batch dimensions added."""
    tensors = []
    for shape in shapes:
        batched_shape = (*batch_sizes, *shape)
        tensor = torch.randn(batched_shape, dtype=dtype, device=device)
        tensors.append(tensor)
    return tuple(tensors)


def _make_reference_shapes(
    shapes: list[tuple[int, ...]],
    *,
    batch_sizes: tuple[int, ...] = (),
) -> list[tuple[int, ...]]:
    return [(*batch_sizes, *shape) for shape in shapes]


@pytest.mark.parametrize(("cls", "params"), MODELS.items())
def test_all_models(cls: type[nn.Module], params: dict) -> None:
    r"""Check if initializations, forward and backward runs for all selected models."""
    logger = logging.getLogger(f"{__name__}/{cls.__name__}")
    logger.info("Testing...")
    input_shapes = params["input_shapes"]
    output_shapes = params["output_shapes"]

    for device, batch_sizes in product(DEVICES, BATCH_SIZES):
        logger.info(
            "Testing %s with batch_shape %s",
            device,
            batch_sizes,
        )
        call_args = _make_tensors(
            input_shapes, batch_sizes=batch_sizes, dtype=DTYPE, device=device
        )
        reference_shapes = _make_reference_shapes(
            output_shapes, batch_sizes=batch_sizes
        )

        model = check_initialization(
            cls,
            init_args=params["init_args"],
            init_kwargs=params["init_kwargs"],
        )

        assert_model_ok(
            model,
            call_args=call_args,
            call_kwargs=params["init_kwargs"],
            reference_shapes=reference_shapes,
            device=device,
        )

    logger.info("Model passed all tests!!")
