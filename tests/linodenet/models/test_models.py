#!/usr/bin/env python
r"""Test if model initializations, forward and backward passes."""

import logging
from itertools import product

import torch
from torch import Tensor

import linodenet
from linodenet.config import PROJECT
from linodenet.models import LinearContraction, LinODE, LinODEnet, iResNet, iResNetBlock
from linodenet.models.system import LinODECell
from linodenet.testing import check_class

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)
RESULT_DIR = PROJECT.TEST_RESULTS_PATH / (PROJECT.TEST_RESULTS_PATH / __file__).stem
RESULT_DIR.mkdir(parents=True, exist_ok=True)

linodenet.CONFIG.autojit = False

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

MODELS: dict[type, dict] = {
    LinearContraction: {
        "initialization": (DIM, OUT),
        "input_shapes": ((LEN, DIM),),  # X
        "output_shapes": ((LEN, OUT),),
    },
    iResNetBlock: {
        "initialization": (DIM,),
        "input_shapes": ((LEN, DIM),),  # X
        "output_shapes": ((LEN, DIM),),
    },
    iResNet: {
        "initialization": (DIM,),
        "input_shapes": ((LEN, DIM),),  # X
        "output_shapes": ((LEN, DIM),),
    },
    LinODECell: {
        "initialization": (DIM,),
        "input_shapes": ((), (DIM,)),  # Δt, x0
        "output_shapes": ((DIM,),),
    },
    LinODE: {
        "initialization": (DIM,),
        "input_shapes": ((LEN,), (DIM,)),  # T, x0
        "output_shapes": ((LEN, DIM),),
    },
    LinODEnet: {
        "initialization": (DIM, LAT),
        "input_shapes": ((LEN,), (LEN, DIM)),  # T, X
        "output_shapes": ((LEN, DIM),),
    },
}


def _make_tensors(
    shapes: tuple[tuple[int, ...]],
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


def test_all_models() -> None:
    r"""Check if initializations, forward and backward runs for all selected models."""
    __logger__.info("Testing forward/backward of %s.", set(MODELS))

    for model, params in MODELS.items():
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
            targets = _make_tensors(
                output_shapes, batch_sizes=batch_sizes, dtype=DTYPE, device=device
            )
            check_class(
                model,
                init_args=initialization,
                input_args=inputs,
                reference_outputs=targets,
                device=device,
            )

        LOGGER.info("Model passed all tests!!")

    __logger__.info("Finished testing forward/backward of %s.", set(MODELS))


def _main() -> None:
    test_all_models()


if __name__ == "__main__":
    _main()
