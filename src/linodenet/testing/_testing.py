"""Utility functions for testing."""

__all__ = [
    # functions
    "check_forward",
    "check_backward",
    "check_jit",
    "check_jit_saving_loading",
    "check_function",
    "check_model",
    "check_class",
    # helper functions
    "flatten_nested_tensor",
    "get_device",
    "get_grads",
    "get_parameters",
    "get_norm",
    "make_tensors_parameters",
    "to_device",
    "zero_grad",
]

import logging
import tempfile
import warnings
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from itertools import chain
from typing import Optional, overload

import torch
from torch import Tensor, jit, nn
from typing_extensions import deprecated

from linodenet.constants import EMPTY_MAP
from linodenet.types import Nested, Scalar

__logger__ = logging.getLogger(__name__)


# region utility functions for tensors AND scalars -------------------------------------
@overload
def to_device(
    x: Nested[Tensor | Scalar], /, *, device: torch.device
) -> Nested[Tensor | Scalar]:
    ...


@overload
def to_device(x: nn.Module, /, *, device: torch.device) -> nn.Module:
    ...


def to_device(x, /, *, device):
    """Move a nested tensor to a device."""
    # FIXME: https://github.com/python/cpython/issues/106246. Use match-case when fixed.
    if isinstance(x, nn.Module):
        return x.to(device=device)
    if isinstance(x, Tensor):
        return x.to(device=device)
    if isinstance(x, Scalar):
        return x
    if isinstance(x, Mapping):
        return {key: to_device(val, device=device) for key, val in x.items()}
    if isinstance(x, Iterable):
        return tuple(to_device(item, device=device) for item in x)
    raise TypeError(f"Unsupported input type {type(x)!r}")


def get_device(x: nn.Module | Nested[Tensor | Scalar], /) -> torch.device:
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


def make_tensors_parameters(x: Nested[Tensor | Scalar], /) -> Nested[Tensor | Scalar]:
    """Make tensors parameters."""
    # FIXME: https://github.com/python/cpython/issues/106246. Use match-case when fixed.
    if isinstance(x, Scalar):
        return x
    if isinstance(x, Tensor):
        return nn.Parameter(x) if not isinstance(x, nn.Parameter) else x
    if isinstance(x, Mapping):
        return {key: make_tensors_parameters(val) for key, val in x.items()}
    if isinstance(x, Iterable):
        return tuple(make_tensors_parameters(item) for item in x)
    raise TypeError(f"Unsupported input type {type(x)!r}")


# endregion utility functions for tensors AND scalars ----------------------------------


# region utility functions  for outputs (always tensor) --------------------------------
def flatten_nested_tensor(x: Nested[Tensor], /) -> Tensor:
    r"""Flattens element of general Hilbert space, skips over scalars."""
    match x:
        case Tensor() as tensor:
            return tensor.flatten()
        case Mapping() as mapping:
            return torch.cat([flatten_nested_tensor(val) for val in mapping.values()])
        case Iterable() as iterable:
            return torch.cat([flatten_nested_tensor(item) for item in iterable])
        case _:
            raise TypeError(f"Unsupported input type {type(x)!r}")


def get_parameters(x: nn.Module | Nested[Tensor], /) -> list[nn.Parameter]:
    """Return the parameters of the model / parameters."""
    match x:
        case nn.Module() as model:
            return list(x for x in model.parameters() if x.requires_grad)
        case Tensor() as tensor:
            if tensor.requires_grad:
                return [tensor]
            return []
        case Mapping() as mapping:
            return list(
                chain.from_iterable(get_parameters(item) for item in mapping.values())
            )
        case Iterable() as iterable:
            return list(chain.from_iterable(get_parameters(item) for item in iterable))
        case _:
            raise TypeError(f"Unsupported input type {type(x)!r}")


def get_norm(x: Nested[Tensor], /, *, normalize: bool = True) -> Tensor:
    """Compute the (normalized) 2-norm of a tensor."""
    flattened = flatten_nested_tensor(x)
    if normalize:
        return torch.sqrt(torch.mean(flattened**2))
    return torch.sqrt(torch.sum(flattened**2))


def get_grads(x: nn.Module | Nested[Tensor], /) -> list[Tensor]:
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


def zero_grad(x: nn.Module | Nested[Tensor], /) -> None:
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


def assert_close(
    values: Nested[Tensor],
    reference: Nested[Tensor],
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


# endregion utility functions  for outputs (always tensor) -----------------------------


def check_forward(
    func: Callable[..., Nested[Tensor]],
    /,
    *,
    input_args: Iterable[Nested[Tensor]] = (),
    input_kwargs: Mapping[str, Nested[Tensor]] = EMPTY_MAP,
    reference_outputs: Optional[Nested[Tensor]] = None,
) -> tuple[Nested[Tensor], Nested[Tensor]]:
    """Test a forward pass."""
    try:
        outputs = func(*input_args, **input_kwargs)
    except Exception as exc:
        raise RuntimeError("Forward pass failed!!") from exc

    if reference_outputs is None:
        reference_outputs = outputs
    else:
        assert_close(outputs, reference_outputs)
    return outputs, reference_outputs


def check_backward(
    *,
    outputs: Nested[Tensor],
    parameters: Iterable[nn.Parameter],
    reference_gradients: Optional[Nested[Tensor]] = None,
) -> tuple[list[Tensor], list[Tensor]]:
    """Test a backward pass."""
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

    return gradients, reference_gradients


# fmt: off
@overload
def check_jit(module: nn.Module, /) -> nn.Module: ...
@overload
def check_jit(func: Callable[..., Nested[Tensor]], /) -> Callable[..., Nested[Tensor]]: ...
# fmt: on
def check_jit(module_or_func, /):
    """Test JIT compilation."""
    try:
        scripted = jit.script(module_or_func)
    except Exception as exc:
        raise RuntimeError("Model JIT compilation Failed!") from exc
    return scripted


# fmt: off
@overload
def check_jit_saving_loading(
    module: nn.Module, /, *, device: torch.device
) -> nn.Module: ...
@overload
def check_jit_saving_loading(
    func: Callable[..., Nested[Tensor]], /, *, device: torch.device
) -> Callable[..., Nested[Tensor]]: ...
# fmt: on
def check_jit_saving_loading(scripted, /, *, device: torch.device):
    """Test saving and loading of JIT compiled model."""
    with tempfile.TemporaryFile() as file:
        try:
            jit.save(scripted, file)
            file.seek(0)
        except Exception as exc:
            raise RuntimeError("Model saving failed!") from exc

        try:
            loaded = jit.load(file, map_location=device)
        except Exception as exc:
            raise RuntimeError("Model loading failed!") from exc
    return loaded


def check_initialization(
    module_class: type[nn.Module],
    /,
    *,
    init_args: Iterable[Nested[Tensor | Scalar]] = (),
    init_kwargs: Mapping[str, Nested[Tensor | Scalar]] = EMPTY_MAP,
) -> nn.Module:
    """Test initialization of a module."""
    try:
        module = module_class(*init_args, **init_kwargs)
    except Exception as exc:
        raise RuntimeError("Model initialization failed!") from exc
    return module


def check_optim(
    model: nn.Module,
    /,
    *,
    input_args: Iterable[Nested[Tensor]] = (),
    input_kwargs: Mapping[str, Nested[Tensor]] = EMPTY_MAP,
    niter: int = 3,
) -> None:
    """Check if model can be optimized"""
    # create optimizer
    optim = torch.optim.SGD(params=model.parameters(), lr=0.1)

    # perform iterations
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        outputs = model(*input_args, **input_kwargs)
        r = get_norm(outputs)
        r.backward()
        optim.step()


def check_combined(
    obj: type[nn.Module] | nn.Module | Callable[..., Nested[Tensor]],
    *,
    init_args: Iterable[Nested[Tensor | Scalar]] = (),
    init_kwargs: Mapping[str, Nested[Tensor | Scalar]] = EMPTY_MAP,
    #
    input_args: Iterable[Nested[Tensor]] = (),
    input_kwargs: Mapping[str, Nested[Tensor]] = EMPTY_MAP,
    #
    reference_gradients: Optional[Nested[Tensor]] = None,
    reference_model: Optional[nn.Module] = None,
    reference_outputs: Optional[Nested[Tensor]] = None,
    # extra arguments
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
    make_inputs_parameters: bool = True,
    test_jit: bool = False,
    test_optim: bool = False,
) -> None:
    """Check a module, function or model class."""

    # region get configuration -------------------------------------------------
    pass
    # endregion get configuration ----------------------------------------------

    # region get name and logger -------------------------------------------------------
    match obj:
        case type() if issubclass(obj, nn.Module):
            model_name = obj.__name__
        case nn.Module() as model:
            model_name = model.__class__.__name__
        case Callable() as func:
            model_name = func.__name__
        case _:
            raise TypeError(f"Unsupported type {type(obj)} for `obj`!")

    # initialize logger
    logger = __logger__.getChild(model_name) if logger is None else logger
    # endregion get name and logger ----------------------------------------------------

    # region get initialized model if class --------------------------------------------
    if issubclass(obj, nn.Module):
        model = check_initialization(obj, init_args=init_args, init_kwargs=init_kwargs)
        logger.info(">>> Initialization ✔ ")
    else:
        model = obj
    # endregion get initialized model if class --------------------------------------------

    # region get parameters ------------------------------------------------------------
    if isinstance(model, nn.Module):
        model_parameters = get_parameters(model)
    else:
        model_parameters = []

    # get parameters of input tensors
    if make_inputs_parameters:
        input_args = make_tensors_parameters(input_args)
        input_kwargs = make_tensors_parameters(input_kwargs)
        input_parameters = get_parameters((input_args, input_kwargs))
    else:
        input_parameters = []

    parameters = model_parameters + input_parameters
    # endregion get parameters model ---------------------------------------------------

    # region get reference model -------------------------------------------------------
    if reference_model is not None:
        assert (
            reference_outputs is None and reference_gradients is None
        ), "Cannot specify both reference model and reference outputs/gradients!"

        try:
            reference_model.to(device=device)
            reference_outputs = reference_model(*input_args, **input_kwargs)
            r = get_norm(reference_outputs)
            r.backward()
            reference_gradients = get_grads(reference_model)
            reference_model.zero_grad(set_to_none=True)
        except Exception as exc:
            raise RuntimeError("Reference model failed forward/backward pass!") from exc
        logger.info(">>> Reference model forward/backward ✔ ")
    # endregion get reference model ----------------------------------------------------

    # region change device -------------------------------------------------------------
    if device is None:
        device = get_device(model)

    try:
        model = model.to(device=device)
        input_args = to_device(input_args, device=device)
        if reference_outputs is not None:
            reference_outputs = to_device(reference_outputs, device=device)
        if reference_gradients is not None:
            reference_gradients = to_device(reference_gradients, device=device)
    except Exception as exc:
        raise RuntimeError("Couldn't move model/tensors to device!") from exc

    logger.info(">>> Moved model/tensors to Device ✔ ")
    # endregion change device ----------------------------------------------------------

    # region check forward pass --------------------------------------------------------
    outputs, reference_outputs = check_forward(
        model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        reference_outputs=reference_outputs,
    )
    logger.info(">>> Forward ✔ ")
    # endregion check forward pass -----------------------------------------------------

    # region check backward pass -------------------------------------------------------
    gradients, reference_gradients = check_backward(
        outputs=outputs,
        parameters=parameters,
        reference_gradients=reference_gradients,
    )
    logger.info(">>> Backward ✔ ")
    # endregion check backward pass ----------------------------------------------------

    # region check optimization ------------------------------------------------
    if test_optim:
        assert isinstance(model, nn.Module), "Cannot test optimization of function!"
        # create a clone of the model, inputs and outputs.
        check_optim(
            deepcopy(model),
            input_args=deepcopy(input_args),
            input_kwargs=deepcopy(input_kwargs),
        )
    # endregion check optimization ---------------------------------------------

    # terminate if not testing JIT
    if not test_jit:
        return

    # region check JIT compilation -----------------------------------------------------
    scripted_model = check_jit(model)
    logger.info(">>> JIT-compilation ✔ ")
    # endregion check forward pass -----------------------------------------------------

    # region check scripted forward/backward pass --------------------------------------
    check_combined(
        scripted_model,
        input_args=input_args,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        logger=__logger__.getChild(f"{model_name}@JIT"),
    )
    # endregion check scripted forward/backward pass -----------------------------------

    # region check model saving/loading ------------------------------------------------
    loaded_model = check_jit_saving_loading(scripted_model, device=device)
    logger.info(">>> JIT-loading ✔ ")
    # endregion check model saving/loading ---------------------------------------------

    # region check loaded forward/backward pass ----------------------------------------
    check_combined(
        loaded_model,
        input_args=input_args,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        logger=__logger__.getChild(f"{model_name}@DESERIALIZED"),
    )
    # endregion check loaded forward/backward pass -------------------------------------


@deprecated()
def check_function(
    func: Callable[..., Nested[Tensor]],
    /,
    *,
    input_args: Iterable[Nested[Tensor]] = (),
    input_kwargs: Mapping[str, Nested[Tensor]] = EMPTY_MAP,
    reference_function: Optional[Callable[..., Nested[Tensor]]] = None,
    reference_outputs: Optional[Nested[Tensor]] = None,
    reference_gradients: Optional[Nested[Tensor]] = None,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    make_inputs_parameters: bool = True,
    test_jit: bool = False,
    test_optim: bool = False,
) -> None:
    """Test a function."""
    function_name = func.__name__

    # initialize logger
    if logger is None:
        logger = __logger__.getChild(function_name)

    # region prepare inputs ------------------------------------------------------------
    if isinstance(input_args, Tensor):
        warnings.warn(
            "Got single input tensor, wrapping in tuple!",
            UserWarning,
            stacklevel=2,
        )
        input_args = (input_args,)

    if make_inputs_parameters:
        input_args = make_tensors_parameters(input_args)
        input_kwargs = make_tensors_parameters(input_kwargs)
    # endregion prepare inputs ---------------------------------------------------------

    # region get parameters model ------------------------------------------------------
    parameters = (input_args, input_kwargs)
    # endregion get parameters model ---------------------------------------------------

    # region get reference function ----------------------------------------------------
    if reference_function is not None:
        assert (
            reference_outputs is None and reference_gradients is None
        ), "Cannot specify both reference model and reference outputs/gradients!"

        try:
            reference_outputs = reference_function(*input_args, **input_kwargs)
            r = get_norm(reference_outputs)
            r.backward()
            reference_gradients = get_grads(parameters)
            zero_grad(parameters)
        except Exception as exc:
            raise RuntimeError("Reference model failed forward/backward pass!") from exc
        logger.info(">>> Reference model forward/backward ✔ ")
    # endregion get reference model ----------------------------------------------------

    # region change device -------------------------------------------------------------
    if device is None:
        device = get_device(input_args)

    try:
        input_args = to_device(input_args, device=device)
        if reference_outputs is not None:
            reference_outputs = to_device(reference_outputs, device=device)
        if reference_gradients is not None:
            reference_gradients = to_device(reference_gradients, device=device)
    except Exception as exc:
        raise RuntimeError("Couldn't move model/tensors to device!") from exc

    logger.info(">>> Moved model/tensors to Device ✔ ")
    # endregion change device ----------------------------------------------------------

    # region check forward pass --------------------------------------------------------
    try:
        outputs = func(*input_args, **input_kwargs)
    except Exception as exc:
        raise RuntimeError("Model failed forward pass!") from exc

    if reference_outputs is None:
        reference_outputs = outputs
    else:
        assert_close(outputs, reference_outputs)

    logger.info(">>> Forward ✔ ")
    # endregion check forward pass -----------------------------------------------------

    # region check backward pass -------------------------------------------------------
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
    # endregion check backward pass ----------------------------------------------------

    # region check optimization ------------------------------------------------
    if test_optim:
        # perform a gradient update and a second forward pass.
        raise NotImplementedError
    # endregion check optimization ---------------------------------------------

    # terminate if not testing JIT
    if not test_jit:
        return

    # region check JIT compilation -----------------------------------------------------
    try:
        scripted_function = jit.script(func)
    except Exception as exc:
        raise RuntimeError("Model JIT compilation Failed!") from exc

    logger.info(">>> JIT-compilation ✔ ")
    # endregion check forward pass -----------------------------------------------------

    # region check scripted forward/backward pass --------------------------------------
    check_function(
        scripted_function,
        input_args=input_args,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        logger=__logger__.getChild(f"{function_name}@JIT"),
    )
    # endregion check scripted forward/backward pass -----------------------------------

    # region check model saving/loading ------------------------------------------------
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
    # endregion check model saving/loading ---------------------------------------------

    # region check loaded forward/backward pass ----------------------------------------
    check_function(
        loaded_function,
        input_args=input_args,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        logger=__logger__.getChild(f"{function_name}@JIT"),
    )
    # endregion check loaded forward/backward pass -------------------------------------


@deprecated
def check_model(
    model: nn.Module,
    /,
    *,
    input_args: Iterable[Nested[Tensor]] = (),
    input_kwargs: Mapping[str, Nested[Tensor]] = EMPTY_MAP,
    reference_model: Optional[nn.Module] = None,
    reference_outputs: Optional[Nested[Tensor]] = None,
    reference_gradients: Optional[Nested[Tensor]] = None,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    test_jit: bool = False,
    test_optim: bool = False,
) -> None:
    """Test a model."""
    model_name = model.__class__.__name__

    # initialize logger
    if logger is None:
        logger = __logger__.getChild(model_name)

    # region prepare inputs ------------------------------------------------------------
    if isinstance(input_args, Tensor):
        warnings.warn(
            "Got single input tensor, wrapping in tuple!", UserWarning, stacklevel=2
        )
        input_args = (input_args,)
    # endregion prepare inputs ---------------------------------------------------------

    # region get parameters model ------------------------------------------------------
    parameters = tuple(model.parameters())
    # endregion get parameters model ---------------------------------------------------

    # region get reference model -------------------------------------------------------
    if reference_model is not None:
        assert (
            reference_outputs is None and reference_gradients is None
        ), "Cannot specify both reference model and reference outputs/gradients!"

        try:
            reference_model.to(device=device)
            reference_outputs = reference_model(*input_args, **input_kwargs)
            r = get_norm(reference_outputs)
            r.backward()
            reference_gradients = get_grads(reference_model)
            reference_model.zero_grad(set_to_none=True)
        except Exception as exc:
            raise RuntimeError("Reference model failed forward/backward pass!") from exc
        logger.info(">>> Reference model forward/backward ✔ ")
    # endregion get reference model ----------------------------------------------------

    # region change device -------------------------------------------------------------
    if device is None:
        device = get_device(model)

    try:
        model = model.to(device=device)
        input_args = to_device(input_args, device=device)
        if reference_outputs is not None:
            reference_outputs = to_device(reference_outputs, device=device)
        if reference_gradients is not None:
            reference_gradients = to_device(reference_gradients, device=device)
    except Exception as exc:
        raise RuntimeError("Couldn't move model/tensors to device!") from exc

    logger.info(">>> Moved model/tensors to Device ✔ ")
    # endregion change device ----------------------------------------------------------

    # region check forward pass --------------------------------------------------------
    try:
        outputs = model(*input_args, **input_kwargs)
    except Exception as exc:
        raise RuntimeError("Model failed forward pass!") from exc

    if reference_outputs is None:
        reference_outputs = outputs
    else:
        assert_close(outputs, reference_outputs)

    logger.info(">>> Forward ✔ ")
    # endregion check forward pass -----------------------------------------------------

    # region check backward pass -------------------------------------------------------
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
    # endregion check backward pass ----------------------------------------------------

    # region check optimization ------------------------------------------------
    if test_optim:
        # perform a gradient update and a second forward pass.
        optim = torch.optim.SGD(params=parameters, lr=0.1)
        optim.step()
        raise NotImplementedError
    # endregion check optimization ---------------------------------------------

    # terminate if not testing JIT
    if not test_jit:
        return

    # region check JIT compilation -----------------------------------------------------
    try:
        scripted_model = jit.script(model)
    except Exception as exc:
        raise RuntimeError("Model JIT compilation Failed!") from exc

    logger.info(">>> JIT-compilation ✔ ")
    # endregion check forward pass -----------------------------------------------------

    # region check scripted forward/backward pass --------------------------------------
    check_model(
        scripted_model,
        input_args=input_args,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        logger=__logger__.getChild(f"{model_name}@JIT"),
    )
    # endregion check scripted forward/backward pass -----------------------------------

    # region check model saving/loading ------------------------------------------------
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
    # endregion check model saving/loading ---------------------------------------------

    # region check loaded forward/backward pass ----------------------------------------
    check_model(
        loaded_model,
        input_args=input_args,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        logger=__logger__.getChild(f"{model_name}@JIT"),
    )
    # endregion check loaded forward/backward pass -------------------------------------


@deprecated
def check_class(
    model_class: type[nn.Module],
    *,
    init_args: Iterable[Nested[Tensor | Scalar]] = (),
    init_kwargs: Mapping[str, Nested[Tensor | Scalar]] = EMPTY_MAP,
    input_args: Iterable[Nested[Tensor]] = (),
    input_kwargs: Mapping[str, Nested[Tensor]] = EMPTY_MAP,
    reference_gradients: Optional[Nested[Tensor]] = None,
    reference_model: Optional[nn.Module] = None,
    reference_outputs: Optional[Nested[Tensor]] = None,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    test_jit: bool = False,
) -> None:
    """Test a model class."""
    class_name = model_class.__name__

    # initialize logger
    if logger is None:
        logger = __logger__.getChild(class_name)

    # region initialize model ----------------------------------------------------------
    try:
        model = model_class(*init_args, **init_kwargs)
    except Exception as exc:
        raise RuntimeError("Model initialization failed!") from exc
    logger.info(">>> Initialization ✔ ")
    # endregion initialize model -------------------------------------------------------

    # test with initialized model
    check_model(
        model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        reference_model=reference_model,
        reference_outputs=reference_outputs,
        reference_gradients=reference_gradients,
        logger=logger,
        device=device,
        test_jit=test_jit,
    )
