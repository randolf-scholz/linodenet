"""Utility functions for testing."""

__all__ = ["test_model", "test_model_class"]

import logging
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional, TypeAlias

import torch
from torch import Tensor, jit, nn
from torch.nn.functional import mse_loss

__logger__ = logging.getLogger(__name__)
DTYPE = torch.float32
DEVICES = [torch.device("cpu")]


NestedTensor: TypeAlias = (
    Tensor | Mapping[Any, "NestedTensor"] | Iterable["NestedTensor"]
)


def flatten_nested_tensor(x: NestedTensor, /) -> Tensor:
    r"""Flattens element of general Hilbert space."""
    if isinstance(x, Tensor):
        return torch.flatten(x)
    if isinstance(x, Mapping):
        return torch.cat([flatten_nested_tensor(x) for x in x.values()])
    if isinstance(x, Iterable):
        return torch.cat([flatten_nested_tensor(x) for x in x])
    raise ValueError(f"{x=} not understood")


def to_device_nested_tensor(
    x: NestedTensor, /, *, device: torch.device
) -> NestedTensor:
    """Move a nested tensor to a device."""
    if isinstance(x, Tensor):
        return x.to(device=device)
    if isinstance(x, Mapping):
        return {key: to_device_nested_tensor(x[key], device=device) for key in x}
    if isinstance(x, Iterable):
        return [to_device_nested_tensor(x, device=device) for x in x]
    raise ValueError(f"{x=} not understood")


def get_norm(x: NestedTensor, /, normalize: bool = True) -> Tensor:
    """Compute the (normalized) 2-norm of a tensor."""
    flattened = flatten_nested_tensor(x)
    if normalize:
        return torch.sqrt(torch.mean(flattened**2))
    return torch.sqrt(torch.sum(flattened**2))


def get_grads(model: nn.Module) -> list[Tensor]:
    """Return a cloned detached copy of the gradients of the model."""
    return [
        param.grad.clone().detach()
        for param in model.parameters()
        if param.grad is not None
    ]


def get_device(model: nn.Module, /) -> torch.device:
    return next(t.device for t in model.parameters())


def assert_close(
    values: NestedTensor,
    reference: NestedTensor,
    /,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert that outputs and targets are close."""
    match values:
        case Tensor():
            assert isinstance(reference, Tensor)
            assert torch.allclose(values, reference, rtol=rtol, atol=atol), (
                values,
                reference,
            )
        case Mapping():
            assert isinstance(reference, Mapping)
            assert values.keys() == reference.keys()
            for key in values.keys():
                assert_close(values[key], reference[key], rtol=rtol, atol=atol)
        case Iterable():
            assert isinstance(reference, Iterable)
            for output, target in zip(values, reference, strict=True):
                assert_close(output, target, rtol=rtol, atol=atol)
        case _:
            raise TypeError(f"Unsupported type {type(values)} for `outputs`!")


def test_model(
    model: nn.Module,
    /,
    *,
    inputs: NestedTensor,
    reference_outputs: Optional[NestedTensor] = None,
    reference_gradients: Optional[NestedTensor] = None,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    test_jit: bool = False,
) -> None:
    """Test a model."""
    model_name = model.__class__.__name__

    # initialize logger
    if logger is None:
        logger = __logger__.getChild(model_name)

    # region change device -----------------------------------------------------
    if device is None:
        device = get_device(model)

    try:
        model = model.to(device)
        inputs = to_device_nested_tensor(inputs, device=device)
        if reference_outputs is not None:
            reference_outputs = to_device_nested_tensor(
                reference_outputs, device=device
            )
        if reference_gradients is not None:
            reference_gradients = to_device_nested_tensor(
                reference_gradients, device=device
            )
    except Exception as exc:
        raise RuntimeError("Couldn't move model/tensors to device!") from exc

    logger.info(">>> Moved model/tensors to Device ✔ ")
    # endregion change device --------------------------------------------------

    # region check forward pass ------------------------------------------------
    try:
        outputs = model(inputs)
    except Exception as exc:
        raise RuntimeError("Model failed forward pass!") from exc

    if reference_outputs is None:
        reference_outputs = outputs
    else:
        assert_close(outputs, reference_outputs)

    logger.info(">>> Forward ✔ ")
    # endregion check forward pass ---------------------------------------------

    # region check backward pass -----------------------------------------------
    try:
        r = get_norm(outputs)
        r.backward()
        gradients = get_grads(model)
        model.zero_grad(set_to_none=True)
    except Exception as exc:
        raise RuntimeError("Model failed backward pass!") from exc

    if reference_gradients is None:
        reference_gradients = gradients
    else:
        assert_close(gradients, reference_gradients)

    logger.info(">>> Backward ✔ ")
    # endregion check backward pass --------------------------------------------

    # terminate if not testing JIT
    if not test_jit:
        return

    # region check JIT compilation ---------------------------------------------
    try:
        scripted_model = jit.script(model)
    except Exception as exc:
        raise RuntimeError("Model JIT compilation Failed!") from exc

    logger.info(">>> JIT-compilation ✔ ")
    # endregion check forward pass ---------------------------------------------

    # region check scripted forward/backward pass ------------------------------
    test_model(
        scripted_model,
        inputs=inputs,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        logger=__logger__.getChild(f"{model_name}@JIT"),
    )
    # endregion check scripted forward/backward pass ---------------------------

    # region check model saving/loading ----------------------------------------
    with tempfile.TemporaryFile() as file:
        try:
            jit.save(scripted_model, file)
            file.seek(0)
        except Exception as exc:
            raise RuntimeError("Model saving failed!") from exc
        logger.info(">>> JIT-saving ✔ ")

        try:
            loaded_model = jit.load(file, map_location=device)
        except Exception as exc:
            raise RuntimeError("Model loading failed!") from exc
        logger.info(">>> JIT-loading ✔ ")
    # endregion check model saving/loading -------------------------------------

    # region check loaded forward/backward pass ------------------------------
    test_model(
        loaded_model,
        inputs=inputs,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        logger=__logger__.getChild(f"{model_name}@JIT"),
    )
    # endregion check loaded forward/backward pass ---------------------------


def test_model_class(
    Model: type[nn.Module],
    *,
    initialization: tuple[int, ...],
    inputs: tuple[Tensor, ...],
    targets: tuple[Tensor, ...],
    device: torch.device = DEVICES[0],
) -> None:
    """Test a model class."""
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
    LOGGER.info(">>> INITIALIZATION ✔ ")

    try:  # check JIT-compatibility
        LOGGER.info(">>> JIT-COMPILATION TEST")
        model = jit.script(model)
    except Exception as E:
        raise RuntimeError(err_str("JIT-compilation")) from E
    LOGGER.info(">>> JIT-compilation ✔ ")

    try:  # check forward
        LOGGER.info(
            ">>> FORWARD with input shapes %s", [tuple(x.shape) for x in inputs]
        )
        outputs = model(*inputs)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    except Exception as E:
        raise RuntimeError(err_str("forward pass")) from E
    assert all(output.shape == target.shape for output, target in zip(outputs, targets))
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
    LOGGER.info(">>> BACKWARD ✔ ")

    try:  # check model saving
        LOGGER.info(">>> CHECKPOINTING TEST")
        filepath = Path.cwd().joinpath(f"model_checkpoints/{Model.__name__}.pt")
        filepath.parent.mkdir(exist_ok=True)
        jit.save(model, filepath)
        LOGGER.info(">>> Model saved successfully ✔ ")
        model2 = jit.load(filepath)
        LOGGER.info(">>> Model loaded successfully ✔ ")

        residual = flatten_nested_tensor(model(*inputs)) - flatten_nested_tensor(
            model2(*inputs)
        )
        assert (residual == 0.0).all(), f"{torch.mean(residual**2)=}"
        LOGGER.info(">>> Loaded Model produces equivalent outputs ✔ ")
    except Exception as E:
        raise RuntimeError(err_str("checkpointing")) from E
    LOGGER.info(">>> CHECKPOINTING ✔ ")
