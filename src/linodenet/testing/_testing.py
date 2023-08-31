"""Utility functions for testing."""

__all__ = [
    # functions
    "check_function",
    "check_model",
    "check_class",
    # helper functions
    "flatten_nested_tensor",
    "get_device",
    "get_grads",
    "get_norm",
    "make_tensors_parameters",
    "to_device",
    "zero_grad",
]

import logging
import tempfile
import warnings
from collections.abc import Callable, Iterable, Mapping
from itertools import chain
from pathlib import Path
from typing import Optional, overload

import torch
from torch import Tensor, jit, nn
from torch.nn.functional import mse_loss

from linodenet.utils.constants import EMPTY_MAP
from linodenet.utils.types import NestedTensor

__logger__ = logging.getLogger(__name__)


DTYPE = torch.float32
DEVICES = [torch.device("cpu")]


def flatten_nested_tensor(x: NestedTensor, /) -> Tensor:
    r"""Flattens element of general Hilbert space."""
    match x:
        case Tensor() as tensor:
            return tensor.flatten()
        case Mapping() as mapping:
            return torch.cat([flatten_nested_tensor(val) for val in mapping.values()])
        case Iterable() as iterable:
            return torch.cat([flatten_nested_tensor(item) for item in iterable])
        case _:
            raise TypeError(f"Unsupported input type {type(x)!r}")


@overload
def to_device(x: NestedTensor, /, *, device: torch.device) -> NestedTensor:
    ...


@overload
def to_device(x: nn.Module, /, *, device: torch.device) -> nn.Module:
    ...


def to_device(x, /, *, device):
    """Move a nested tensor to a device."""
    match x:
        case nn.Module() as model:
            return model.to(device=device)
        case Tensor() as tensor:
            return tensor.to(device=device)
        case Mapping() as mapping:
            return {key: to_device(val, device=device) for key, val in mapping.items()}
        case Iterable() as iterable:
            return tuple(to_device(x, device=device) for x in iterable)
        case _:
            raise TypeError(f"Unsupported input type {type(x)!r}")


def get_norm(x: NestedTensor, /, normalize: bool = True) -> Tensor:
    """Compute the (normalized) 2-norm of a tensor."""
    flattened = flatten_nested_tensor(x)
    if normalize:
        return torch.sqrt(torch.mean(flattened**2))
    return torch.sqrt(torch.sum(flattened**2))


def get_grads(x: nn.Module | NestedTensor, /) -> list[Tensor]:
    """Return a cloned detached copy of the gradients of the model / parameters."""
    match x:
        case nn.Module() as model:
            return [
                param.grad.clone().detach()
                for param in model.parameters()
                if param.grad is not None
            ]
        case Tensor() as tensor:
            if tensor.requires_grad and tensor.grad is not None:
                return [tensor.grad.clone().detach()]
            return []
        case Mapping() as mapping:
            return list(
                chain.from_iterable(get_grads(item) for item in mapping.values())
            )
        case Iterable() as iterable:
            return list(chain.from_iterable(get_grads(item) for item in iterable))
        case _:
            raise TypeError(f"Unsupported input type {type(x)!r}")


def get_device(x: nn.Module | NestedTensor, /) -> torch.device:
    """Return the device of the model / parameters."""
    match x:
        case nn.Module() as model:
            return next(t.device for t in model.parameters())
        case Tensor() as tensor:
            return tensor.device
        case Mapping() as mapping:
            return get_device(next(iter(mapping.values())))
        case Iterable() as iterable:
            return get_device(next(iter(iterable)))
        case _:
            raise TypeError(f"Unsupported input type {type(x)!r}")


def zero_grad(x: nn.Module | NestedTensor, /) -> None:
    """Sets gradients of the model / parameters to None."""
    match x:
        case nn.Module() as model:
            model.zero_grad(set_to_none=True)
        case Tensor() as tensor:
            if tensor.requires_grad:
                tensor.grad = None
        case Mapping() as mapping:
            for item in mapping.values():
                zero_grad(item)
        case Iterable() as iterable:
            for item in iterable:
                zero_grad(item)
        case _:
            raise TypeError(f"Unsupported input type {type(x)!r}")


def make_tensors_parameters(x: NestedTensor, /) -> NestedTensor:
    """Make tensors parameters."""
    match x:
        case Tensor() as tensor:
            if not isinstance(tensor, nn.Parameter):
                tensor = nn.Parameter(tensor)
            return tensor
        case Mapping() as mapping:
            return {key: make_tensors_parameters(val) for key, val in mapping.items()}
        case Iterable() as iterable:
            return tuple(make_tensors_parameters(item) for item in iterable)
        case _:
            raise TypeError(f"Unsupported input type {type(x)!r}")


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


def check_model(
    model: nn.Module,
    /,
    *,
    inputs: Iterable[NestedTensor] = (),
    input_kwargs: Mapping[str, NestedTensor] = EMPTY_MAP,
    reference_model: Optional[nn.Module] = None,
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

    # region prepare inputs ----------------------------------------------------
    if isinstance(inputs, Tensor):
        warnings.warn(
            "Got single input tensor, wrapping in tuple!", UserWarning, stacklevel=2
        )
        inputs = (inputs,)
    # endregion prepare inputs -------------------------------------------------

    # region get reference model -----------------------------------------------
    if reference_model is not None:
        assert (
            reference_outputs is None and reference_gradients is None
        ), "Cannot specify both reference model and reference outputs/gradients!"

        try:
            reference_model.to(device=device)
            reference_outputs = reference_model(*inputs, **input_kwargs)
            r = get_norm(reference_outputs)
            r.backward()
            reference_gradients = get_grads(reference_model)
            reference_model.zero_grad(set_to_none=True)
        except Exception as exc:
            raise RuntimeError("Reference model failed forward/backward pass!") from exc
        logger.info(">>> Reference model forward/backward ✔ ")
    # endregion get reference model --------------------------------------------

    # region change device -----------------------------------------------------
    if device is None:
        device = get_device(model)

    try:
        model = model.to(device=device)
        inputs = to_device(inputs, device=device)
        if reference_outputs is not None:
            reference_outputs = to_device(reference_outputs, device=device)
        if reference_gradients is not None:
            reference_gradients = to_device(reference_gradients, device=device)
    except Exception as exc:
        raise RuntimeError("Couldn't move model/tensors to device!") from exc

    logger.info(">>> Moved model/tensors to Device ✔ ")
    # endregion change device --------------------------------------------------

    # region check forward pass ------------------------------------------------
    try:
        outputs = model(*inputs, **input_kwargs)
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
    check_model(
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
    check_model(
        loaded_model,
        inputs=inputs,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        logger=__logger__.getChild(f"{model_name}@JIT"),
    )
    # endregion check loaded forward/backward pass ---------------------------


def check_function(
    func: Callable[..., NestedTensor],
    /,
    *,
    inputs: Iterable[NestedTensor] = (),
    input_kwargs: Mapping[str, NestedTensor] = EMPTY_MAP,
    reference_function: Optional[Callable[..., NestedTensor]] = None,
    reference_outputs: Optional[NestedTensor] = None,
    reference_gradients: Optional[NestedTensor] = None,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    make_inputs_parameters: bool = True,
    test_jit: bool = False,
) -> None:
    """Test a function."""
    function_name = func.__name__

    # initialize logger
    if logger is None:
        logger = __logger__.getChild(function_name)

    # region prepare inputs ----------------------------------------------------
    if isinstance(inputs, Tensor):
        warnings.warn(
            "Got single input tensor, wrapping in tuple!",
            UserWarning,
            stacklevel=2,
        )
        inputs = (inputs,)

    if make_inputs_parameters:
        inputs = make_tensors_parameters(inputs)

    parameters = (inputs, input_kwargs)
    # endregion prepare inputs -------------------------------------------------

    # region get reference function --------------------------------------------
    if reference_function is not None:
        assert (
            reference_outputs is None and reference_gradients is None
        ), "Cannot specify both reference model and reference outputs/gradients!"

        try:
            reference_outputs = reference_function(*inputs, **input_kwargs)
            r = get_norm(reference_outputs)
            r.backward()
            reference_gradients = get_grads(parameters)
            zero_grad(parameters)
        except Exception as exc:
            raise RuntimeError("Reference model failed forward/backward pass!") from exc
        logger.info(">>> Reference model forward/backward ✔ ")
    # endregion get reference model --------------------------------------------

    # region change device -----------------------------------------------------
    if device is None:
        device = get_device(inputs)

    try:
        inputs = to_device(inputs, device=device)
        if reference_outputs is not None:
            reference_outputs = to_device(reference_outputs, device=device)
        if reference_gradients is not None:
            reference_gradients = to_device(reference_gradients, device=device)
    except Exception as exc:
        raise RuntimeError("Couldn't move model/tensors to device!") from exc

    logger.info(">>> Moved model/tensors to Device ✔ ")
    # endregion change device --------------------------------------------------

    # region check forward pass ------------------------------------------------
    try:
        outputs = func(*inputs, **input_kwargs)
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
        gradients = get_grads(parameters)
        zero_grad(parameters)
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
        scripted_function = jit.script(func)
    except Exception as exc:
        raise RuntimeError("Model JIT compilation Failed!") from exc

    logger.info(">>> JIT-compilation ✔ ")
    # endregion check forward pass ---------------------------------------------

    # region check scripted forward/backward pass ------------------------------
    check_function(
        scripted_function,
        inputs=inputs,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        logger=__logger__.getChild(f"{function_name}@JIT"),
    )
    # endregion check scripted forward/backward pass ---------------------------

    # region check model saving/loading ----------------------------------------
    with tempfile.TemporaryFile() as file:
        try:
            jit.save(scripted_function, file)
            file.seek(0)
        except Exception as exc:
            raise RuntimeError("Model saving failed!") from exc
        logger.info(">>> JIT-saving ✔ ")

        try:
            loaded_function = jit.load(file, map_location=device)
        except Exception as exc:
            raise RuntimeError("Model loading failed!") from exc
        logger.info(">>> JIT-loading ✔ ")
    # endregion check model saving/loading -------------------------------------

    # region check loaded forward/backward pass ------------------------------
    check_function(
        loaded_function,
        inputs=inputs,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        logger=__logger__.getChild(f"{function_name}@JIT"),
    )
    # endregion check loaded forward/backward pass ---------------------------


def check_class(
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
