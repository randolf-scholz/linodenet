"""Utility functions for testing."""

__all__ = [
    # check functions
    "check_combined",
    "check_function",
    "check_model",
    "check_class",
    # helper functions
    "check_backward",
    "check_forward",
    "check_initialization",
    "check_jit",
    "check_jit_scripting",
    "check_jit_serialization",
    "check_optim",
    # helper functions
    "flatten_nested_tensor",
    "get_device",
    "get_grads",
    "get_norm",
    "get_parameters",
    "get_shapes",
    "iter_parameters",
    "iter_tensors",
    "make_tensors_parameters",
    "to_device",
    "zero_grad",
]

import logging
import tempfile
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from itertools import chain
from typing import Any, Optional, TypeAlias, overload

import torch
from torch import Tensor, jit, nn
from typing_extensions import deprecated

from linodenet.constants import EMPTY_MAP
from linodenet.testing._utils import assert_close
from linodenet.types import Device, Nested, Scalar, T, callable_var, module_var

__logger__ = logging.getLogger(__name__)
Tree: TypeAlias = Nested[Tensor | Scalar]
Func: TypeAlias = Callable[..., Nested[Tensor]]


# region utility functions for tensors AND scalars -------------------------------------\
def get_device(x: nn.Module | Tree, /) -> torch.device:
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


@overload
def to_device(
    x: module_var, /, *, device: str | torch.device = "cpu"
) -> module_var: ...
@overload
def to_device(x: Tensor, /, *, device: str | torch.device = "cpu") -> Tensor: ...
@overload
def to_device(x: Scalar, /, *, device: str | torch.device = "cpu") -> Scalar: ...  # type: ignore[misc]
@overload
def to_device(
    x: Mapping[str, T], /, *, device: str | torch.device = "cpu"
) -> dict[str, T]: ...
@overload
def to_device(
    x: Sequence[T], /, *, device: str | torch.device = "cpu"
) -> tuple[T, ...]: ...
def to_device(x: Any, /, *, device: str | torch.device = "cpu") -> Any:
    """Move a nested tensor to a device."""
    match x:
        case Tensor() as tensor:
            return tensor.to(device=torch.device(device))
        case nn.Module() as module:
            return module.to(device=torch.device(device))
        case None | bool() | int() | float() | str() as scalar:  # Scalar
            # FIXME: https://github.com/python/cpython/issues/106246
            return scalar
        case Mapping() as mapping:
            return {key: to_device(val, device=device) for key, val in mapping.items()}
        case Iterable() as iterable:
            return tuple(to_device(item, device=device) for item in iterable)
    raise TypeError(f"Unsupported input type {type(x)!r}")


def iter_tensors(x: nn.Module | Tree, /) -> Iterator[Tensor]:
    """Iterate over the parameters of the model / parameters."""
    match x:
        case Tensor() as tensor:
            yield tensor
        case nn.Module() as module:
            yield from module.parameters()
        case None | bool() | int() | float() | str():  # Scalar
            # FIXME: https://github.com/python/cpython/issues/106246
            pass
        case Mapping() as mapping:
            yield from chain.from_iterable(iter_tensors(v) for v in mapping.values())
        case Iterable() as iterable:
            yield from chain.from_iterable(iter_tensors(item) for item in iterable)
        case _:
            raise TypeError(f"Unsupported input type {type(x)!r}")


def iter_parameters(x: nn.Module | Tree, /) -> Iterator[nn.Parameter]:
    """Iterate over the parameters of the model / parameters."""
    for w in iter_tensors(x):
        if isinstance(w, nn.Parameter):
            yield w


def get_parameters(x: nn.Module | Tree, /) -> list[Tensor]:
    """Return the parameters of the model / parameters."""
    return [w for w in iter_tensors(x) if w.requires_grad]


def zero_grad(x: nn.Module | Tree, /) -> None:
    """Sets gradients of the model / parameters to None."""
    if isinstance(x, nn.Module):
        x.zero_grad(set_to_none=True)
        return

    for w in iter_tensors(x):
        if w.requires_grad:
            w.grad = None


# endregion utility functions for tensors AND scalars ----------------------------------


# region utility functions  for outputs (always tensor) --------------------------------
def flatten_nested_tensor(x: nn.Module | Tree, /) -> Tensor:
    r"""Flattens element of general Hilbert space, skips over scalars."""
    return torch.cat([x.flatten() for x in iter_tensors(x)])


def get_shapes(x: nn.Module | Tree, /) -> list[tuple[int, ...]]:
    """Return the shapes of the tensors."""
    return [item.shape for item in iter_tensors(x)]


def get_grads(x: nn.Module | Tree, /) -> list[Tensor]:
    """Return a cloned detached copy of the gradients."""
    return [
        w.grad.clone().detach()
        for w in iter_tensors(x)
        if w.requires_grad and w.grad is not None
    ]


def get_norm(x: Nested[Tensor], /, *, normalize: bool = True) -> Tensor:
    """Compute the (normalized) 2-norm of a tensor."""
    flattened = flatten_nested_tensor(x)
    if normalize:
        return torch.sqrt(torch.mean(flattened**2))
    return torch.sqrt(torch.sum(flattened**2))


@overload
def make_tensors_parameters(x: Tensor, /) -> nn.Parameter: ...
@overload
def make_tensors_parameters(x: Scalar, /) -> Scalar: ...  # type: ignore[misc]
@overload
def make_tensors_parameters(x: Mapping[str, T], /) -> dict[str, T]: ...
@overload
def make_tensors_parameters(x: Sequence[T], /) -> tuple[T, ...]: ...
def make_tensors_parameters(x, /):
    """Make tensors parameters."""
    # FIXME: https://github.com/python/cpython/issues/106246. Use match-case when fixed.
    if isinstance(x, Tensor):
        return nn.Parameter(x) if not isinstance(x, nn.Parameter) else x
    if isinstance(x, Scalar):  # type: ignore[misc, arg-type]
        return x
    if isinstance(x, Mapping):
        return {key: make_tensors_parameters(val) for key, val in x.items()}
    if isinstance(x, Iterable):
        return tuple(make_tensors_parameters(item) for item in x)
    raise TypeError(f"Unsupported input type {type(x)!r}")


# endregion utility functions  for outputs (always tensor) -----------------------------


# region check helper functions --------------------------------------------------------
def check_forward(
    func: Func,
    /,
    *,
    input_args: Sequence[Tree] = (),
    input_kwargs: Mapping[str, Tree] = EMPTY_MAP,
    reference_outputs: Optional[Nested[Tensor]] = None,
    reference_shapes: Optional[Sequence[tuple[int, ...]]] = None,
) -> tuple[Nested[Tensor], Nested[Tensor], list[tuple[int, ...]]]:
    """Test a forward pass."""
    try:
        outputs = func(*input_args, **input_kwargs)
        output_shapes = get_shapes(outputs)
    except Exception as exc:
        raise RuntimeError("Forward pass failed!!") from exc

    # validate shapes
    if reference_shapes is None:
        reference_shapes = output_shapes
    else:
        assert isinstance(
            reference_shapes, list
        ), "reference_shapes must be a list of integer tuples!"
    assert reference_shapes == output_shapes, f"{reference_shapes=} {output_shapes=}"

    # validate values
    if reference_outputs is None:
        reference_outputs = outputs
    assert_close(outputs, reference_outputs)

    return outputs, reference_outputs, reference_shapes


def check_backward(
    *,
    outputs: Nested[Tensor],
    parameters: Sequence[Tensor],
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
        ref_grads = deepcopy(gradients)
    else:
        ref_grads = list(iter_tensors(reference_gradients))

    # check gradients
    assert_close(gradients, ref_grads)

    return gradients, ref_grads


@overload
def check_jit_scripting(module: nn.Module, /) -> nn.Module: ...
@overload
def check_jit_scripting(func: Func, /) -> Func: ...
def check_jit_scripting(module_or_func, /):
    """Test JIT compilation."""
    try:
        scripted = jit.script(module_or_func)
    except Exception as exc:
        raise RuntimeError("Model JIT compilation Failed!") from exc
    return scripted


@overload
def check_jit_serialization(
    module: nn.Module, /, *, device: Device = ...
) -> nn.Module: ...
@overload
def check_jit_serialization(func: Func, /, *, device: Device = ...) -> Func: ...
def check_jit_serialization(scripted, /, *, device=None):
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


@overload
def check_jit(module: nn.Module, /, *, device: Device = ...) -> nn.Module: ...
@overload
def check_jit(func: Func, /, *, device: Device = ...) -> Func: ...
def check_jit(module_or_func, /, *, device=None):
    """Test JIT compilation+serialization."""
    # check if scripting and serialization works
    scripted = check_jit_scripting(module_or_func)
    loaded = check_jit_serialization(scripted, device=device)
    return loaded


@overload
def check_initialization(
    obj: type[module_var],
    /,
    *,
    init_args: Sequence[Tree] = (),
    init_kwargs: Mapping[str, Tree] = EMPTY_MAP,
) -> module_var: ...
@overload
def check_initialization(
    obj: module_var,
    /,
    *,
    init_args: Sequence[Tree] = (),
    init_kwargs: Mapping[str, Tree] = EMPTY_MAP,
) -> module_var: ...
@overload
def check_initialization(
    obj: callable_var,
    /,
    *,
    init_args: Sequence[Tree] = (),
    init_kwargs: Mapping[str, Tree] = EMPTY_MAP,
) -> callable_var: ...
def check_initialization(obj, /, *, init_args=(), init_kwargs=EMPTY_MAP):
    """Test initialization of a module."""
    if not isinstance(obj, type):
        return obj
    if issubclass(obj, nn.Module):
        try:
            module = obj(*init_args, **init_kwargs)
        except Exception as exc:
            raise RuntimeError("Model initialization failed!") from exc
    else:
        raise TypeError(f"Unsupported type {type(obj)} for `obj`!")

    return module


def check_optim(
    model: nn.Module,
    /,
    *,
    input_args: Sequence[Tree] = (),
    input_kwargs: Mapping[str, Tree] = EMPTY_MAP,
    niter: int = 4,
) -> None:
    """Check if model can be optimized."""
    with torch.no_grad():
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        original_outputs = model(*input_args, **input_kwargs)
        original_loss = get_norm(original_outputs)
        # original_params = [w.clone().detach() for w in model.parameters()]

    # perform iterations
    for _ in range(niter):
        model.zero_grad(set_to_none=True)
        outputs = model(*input_args, **input_kwargs)
        loss = get_norm(outputs)
        assert loss.isfinite()
        loss.backward()
        optimizer.step()

    assert loss < original_loss


# endregion check helper functions -----------------------------------------------------


def check_combined(
    obj: type[nn.Module] | nn.Module | Func,
    *,
    init_args: Sequence[Any] = (),
    init_kwargs: Mapping[str, Any] = EMPTY_MAP,
    #
    input_args: Sequence[Tree] = (),
    input_kwargs: Mapping[str, Tree] = EMPTY_MAP,
    #
    reference_gradients: Optional[Nested[Tensor]] = None,
    reference_model: Optional[nn.Module] = None,
    reference_shapes: Optional[list[tuple[int, ...]]] = None,
    reference_outputs: Optional[Nested[Tensor]] = None,
    # extra arguments
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
    make_inputs_parameters: bool = True,
    test_jit: bool = True,
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
        case Callable() as func:  # type: ignore[misc]
            model_name = func.__name__  # type: ignore[unreachable]
        case _:
            raise TypeError(f"Unsupported type {type(obj)} for `obj`!")

    # initialize logger
    logger = __logger__.getChild(model_name) if logger is None else logger
    # endregion get name and logger ----------------------------------------------------

    # region get initialized model if class --------------------------------------------
    model = check_initialization(obj, init_args=init_args, init_kwargs=init_kwargs)
    logger.info(">>> Initialization ✔ ")
    # endregion get initialized model if class --------------------------------------------

    # region get parameters ------------------------------------------------------------
    if isinstance(model, nn.Module):
        model_parameters = get_parameters(model)
    else:
        model_parameters = []  # type: ignore[unreachable]

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
            assert reference_outputs is not None
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
    outputs, reference_outputs, reference_shapes = check_forward(
        model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        reference_outputs=reference_outputs,
        reference_shapes=reference_shapes,
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
    scripted_model = check_jit_scripting(model)
    logger.info(">>> JIT-compilation ✔ ")
    # endregion check forward pass -----------------------------------------------------

    # region check scripted forward/backward pass --------------------------------------
    check_combined(
        scripted_model,
        input_args=input_args,
        reference_outputs=reference_outputs,
        reference_shapes=reference_shapes,
        reference_gradients=reference_gradients,
        logger=__logger__.getChild(f"{model_name}@JIT"),
        test_jit=False,
    )
    # endregion check scripted forward/backward pass -----------------------------------

    # region check model saving/loading ------------------------------------------------
    loaded_model = check_jit_serialization(scripted_model, device=device)
    logger.info(">>> JIT-loading ✔ ")
    # endregion check model saving/loading ---------------------------------------------

    # region check loaded forward/backward pass ----------------------------------------
    check_combined(
        loaded_model,
        input_args=input_args,
        reference_outputs=reference_outputs,
        reference_shapes=reference_shapes,
        reference_gradients=reference_gradients,
        logger=__logger__.getChild(f"{model_name}@DESERIALIZED"),
        test_jit=False,
    )
    # endregion check loaded forward/backward pass -------------------------------------


@deprecated("Deprecated")
def check_function(
    func: Func,
    /,
    *,
    input_args: Sequence[Tree] = (),
    input_kwargs: Mapping[str, Tree] = EMPTY_MAP,
    #
    reference_function: Optional[Func] = None,
    reference_outputs: Optional[Nested[Tensor]] = None,
    reference_gradients: Optional[Nested[Tensor]] = None,
    #
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
    if isinstance(input_args, Tensor):  # type: ignore[unreachable]
        warnings.warn(  # type: ignore[unreachable]
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


@deprecated("Deprecated")
def check_model(
    model: nn.Module,
    /,
    *,
    input_args: Sequence[Nested[Tensor]] = (),
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
    if isinstance(input_args, Tensor):  # type: ignore[unreachable]
        warnings.warn(  # type: ignore[unreachable]
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


@deprecated("Deprecated")
def check_class(
    model_class: type[nn.Module],
    *,
    init_args: Sequence[Any] = (),
    init_kwargs: Mapping[str, Any] = EMPTY_MAP,
    #
    input_args: Sequence[Nested[Tensor]] = (),
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
