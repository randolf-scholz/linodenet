#!/usr/bin/env python
r"""Test if model initializations, forward and backward passes."""

import logging
from itertools import product
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.functional import mse_loss

import linodenet
from linodenet.models import LinearContraction, LinODE, LinODEnet, iResNet, iResNetBlock
from linodenet.models.system import LinODECell
from linodenet.utils import PROJECT_TEST, flatten_nested_tensor

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)
RESULT_DIR = PROJECT_TEST / "results" / Path(__file__).stem
RESULT_DIR.mkdir(parents=True, exist_ok=True)

linodenet.conf.autojit = False

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


def _test_model(
    Model: type,
    *,
    initialization: tuple[int, ...],
    inputs: tuple[Tensor, ...],
    targets: tuple[Tensor, ...],
    device: torch.device = DEVICES[0],
) -> None:
    LOGGER = __logger__.getChild(Model.__name__)
    LOGGER.info("Testing...")

    def err_str(s: str) -> str:
        return (
            f"{Model=} failed {s} with {initialization=} and "
            f"input shapes {tuple(i.shape for i in inputs)}!"
        )

    try:  # check initialization
        LOGGER.info(">>> INITIALIZATION TEST")
        LOGGER.info(">>> input shapes: %s", initialization)
        model = Model(*initialization)
        model.to(dtype=DTYPE, device=device)
    except Exception as E:
        raise RuntimeError(err_str("initialization")) from E
    else:
        LOGGER.info(">>> INITIALIZATION ✔ ")

    try:  # check JIT-compatibility
        __logger__.info(">>> JIT-COMPILATION TEST")
        model = torch.jit.script(model)
    except Exception as E:
        raise RuntimeError(err_str("JIT-compilation")) from E
    else:
        LOGGER.info(">>> JIT-compilation ✔ ")

    try:  # check forward
        LOGGER.info(
            ">>> FORWARD with input shapes %s", [tuple(x.shape) for x in inputs]
        )
        outputs = model(*inputs)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    except Exception as E:
        raise RuntimeError(err_str("forward pass")) from E
    else:
        assert all(
            output.shape == target.shape for output, target in zip(outputs, targets)
        )
        LOGGER.info(
            ">>> Output shapes %s match with targets!",
            [tuple(x.shape) for x in targets],
        )
        LOGGER.info(">>> FORWARD ✔ ")

    try:  # check backward
        LOGGER.info(">>> BACKWARD TEST")
        losses = [mse_loss(output, target) for output, target in zip(outputs, targets)]
        loss = torch.stack(losses).sum()
        loss.backward()
    except Exception as E:
        raise RuntimeError(err_str("backward pass")) from E
    else:
        LOGGER.info(">>> BACKWARD ✔ ")

    try:  # check model saving
        LOGGER.info(">>> CHECKPOINTING TEST")
        filepath = Path.cwd().joinpath(f"model_checkpoints/{Model.__name__}.pt")
        filepath.parent.mkdir(exist_ok=True)
        torch.jit.save(model, filepath)
        LOGGER.info(">>> Model saved successfully ✔ ")
        model2 = torch.jit.load(filepath)
        LOGGER.info(">>> Model loaded successfully ✔ ")

        residual = flatten_nested_tensor(model(*inputs)) - flatten_nested_tensor(
            model2(*inputs)
        )
        assert (residual == 0.0).all(), f"{torch.mean(residual**2)=}"
        LOGGER.info(">>> Loaded Model produces equivalent outputs ✔ ")
    except Exception as E:
        raise RuntimeError(err_str("checkpointing")) from E
    else:
        LOGGER.info(">>> CHECKPOINTING ✔ ")


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
            _test_model(
                model,
                initialization=initialization,
                inputs=inputs,
                targets=targets,
                device=device,
            )

        LOGGER.info("Model passed all tests!!")

    __logger__.info("Finished testing forward/backward of %s.", MODELS)


def _main() -> None:
    test_all_models()


if __name__ == "__main__":
    _main()
