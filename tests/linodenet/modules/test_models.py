r"""Test if model initializations, forward and backward passes."""

import logging
from itertools import product

import torch
from pytest import mark
from torch import Tensor, nn

from linodenet.config import CONFIG, PROJECT
from linodenet.modules import (
    LinearContraction,
    LinODE,
    LinODEnet,
    iResNet,
    iResNetBlock,
)
from linodenet.modules.system import LinODECell
from linodenet.testing import check_object

__logger__ = logging.getLogger(__name__)
CONFIG.autojit = False
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
        "initialization": (DIM, OUT),
        "input_shapes": [(LEN, DIM)],  # X
        "output_shapes": [(LEN, OUT)],
    },
    iResNetBlock: {
        "initialization": (DIM,),
        "input_shapes": [(LEN, DIM)],  # X
        "output_shapes": [(LEN, DIM)],
    },
    iResNet: {
        "initialization": (DIM,),
        "input_shapes": [(LEN, DIM)],  # X
        "output_shapes": [(LEN, DIM)],
    },
    LinODECell: {
        "initialization": (DIM,),
        "input_shapes": [(), (DIM,)],  # Δt, x0
        "output_shapes": [(DIM,)],
    },
    LinODE: {
        "initialization": (DIM,),
        "input_shapes": [(LEN,), (DIM,)],  # T, x0
        "output_shapes": [(LEN, DIM)],
    },
    LinODEnet: {
        "initialization": (DIM, LAT),
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


@mark.parametrize(("model", "params"), MODELS.items())
def test_all_models(model: type[nn.Module], params: dict) -> None:
    r"""Check if initializations, forward and backward runs for all selected models."""
    LOGGER = __logger__.getChild(model.__name__)
    LOGGER.info("Testing...")
    initialization = params["initialization"]
    input_shapes = params["input_shapes"]
    output_shapes = params["output_shapes"]

    for device, batch_sizes in product(DEVICES, BATCH_SIZES):
        LOGGER.info(
            "Testing %s with batch_shape %s",
            device,
            batch_sizes,
        )
        inputs = _make_tensors(
            input_shapes, batch_sizes=batch_sizes, dtype=DTYPE, device=device
        )
        reference_shapes = _make_reference_shapes(
            output_shapes, batch_sizes=batch_sizes
        )
        check_object(
            model,
            init_args=initialization,
            input_args=inputs,
            reference_shapes=reference_shapes,
            device=device,
        )

    LOGGER.info("Model passed all tests!!")
