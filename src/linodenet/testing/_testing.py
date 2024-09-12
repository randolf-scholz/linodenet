r"""Utility functions for testing."""

__all__ = [
    # check functions
    "check_class",
    "check_function",
    "check_model",
    "check_object",
    # helper functions
    "check_backward",
    "check_forward",
    "check_initialization",
    "check_jit",
    "check_jit_scripting",
    "check_jit_serialization",
    "check_optim",
    # helper functions
    "assert_close",
    "assert_compatible_signature",
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

import inspect
import logging
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from itertools import chain
from typing import Any, Optional, overload

import torch
from torch import Tensor, jit, nn
from torch.nn import Module

from linodenet.constants import EMPTY_MAP
from linodenet.types import DeviceArg, Nested, Scalar

__logger__ = logging.getLogger(__name__)

type Tree = Nested[Tensor | Scalar]
type Func = Callable[..., Nested[Tensor]]


def assert_compatible_signature(func: Callable, reference: Callable) -> None:
    r"""Assert that functions signature is wider than reference."""
    fun_sig = inspect.signature(func)
    ref_sig = inspect.signature(reference)

    for param in ref_sig.parameters:
        assert param in fun_sig.parameters
        assert fun_sig.parameters[param].kind == ref_sig.parameters[param].kind


def assert_close(
    values: Nested[Tensor],
    reference: Nested[Tensor],
    /,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    r"""Assert that outputs and targets are close."""
    match values:
        case Tensor() as tensor:
            assert isinstance(reference, Tensor)
            assert torch.allclose(tensor, reference, rtol=rtol, atol=atol), (
                tensor,
                reference,
            )
        case Mapping() as mapping:
            assert isinstance(reference, Mapping)
            assert mapping.keys() == reference.keys()
            for key in mapping:
                x = mapping[key]
                y = reference[key]
                assert_close(x, y, rtol=rtol, atol=atol)
        case Sequence() as sequence:
            assert isinstance(reference, Sequence)
            for output, target in zip(sequence, reference, strict=True):
                assert_close(output, target, rtol=rtol, atol=atol)
        case _:
            raise TypeError(f"Unsupported type {type(values)} for `outputs`!")


# region utility functions for tensors AND scalars -------------------------------------
def get_device(x: Module | Tree, /) -> torch.device:
    r"""Return the device of the model / parameters."""
    match x:
        case Module() as model:
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
def to_device[M: Module](x: M, /, *, device: DeviceArg = ...) -> M: ...
@overload
def to_device(x: Tensor, /, *, device: DeviceArg = ...) -> Tensor: ...
@overload
def to_device(x: Scalar, /, *, device: DeviceArg = ...) -> Scalar: ...
@overload
def to_device[T](x: Mapping[str, T], /, *, device: DeviceArg = ...) -> dict[str, T]: ...
@overload
def to_device[T](x: Sequence[T], /, *, device: DeviceArg = ...) -> tuple[T, ...]: ...
def to_device(x: Any, /, *, device: DeviceArg = "cpu") -> Any:
    r"""Move a nested tensor to a device."""
    match x:
        case Tensor() as tensor:
            target_device = None if device is None else torch.device(device)
            return tensor.to(device=target_device)
        case Module() as module:
            target_device = None if device is None else torch.device(device)
            return module.to(device=target_device)
        case (None | bool() | int() | float() | str()) as scalar:  # Scalar
            # FIXME: https://github.com/python/cpython/issues/106246
            return scalar
        case Mapping() as mapping:
            return {key: to_device(val, device=device) for key, val in mapping.items()}
        case Iterable() as iterable:
            return tuple(to_device(item, device=device) for item in iterable)
    raise TypeError(f"Unsupported input type {type(x)!r}")


def iter_tensors(x: Module | Tree, /) -> Iterator[Tensor]:
    r"""Iterate over the parameters of the model / parameters."""
    match x:
        case Tensor() as tensor:
            yield tensor
        case Module() as module:
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


def iter_parameters(x: Module | Tree, /) -> Iterator[nn.Parameter]:
    r"""Iterate over the parameters of the model / parameters."""
    for w in iter_tensors(x):
        if isinstance(w, nn.Parameter):
            yield w


def get_parameters(x: Module | Tree, /) -> list[Tensor]:
    r"""Return the parameters of the model / parameters."""
    return [w for w in iter_tensors(x) if w.requires_grad]


def zero_grad(x: Module | Tree, /) -> None:
    r"""Sets gradients of the model / parameters to None."""
    if isinstance(x, Module):
        x.zero_grad(set_to_none=True)
        return

    for w in iter_tensors(x):
        if w.requires_grad:
            w.grad = None


# endregion utility functions for tensors AND scalars ----------------------------------


# region utility functions  for outputs (always tensor) --------------------------------
def flatten_nested_tensor(x: Module | Tree, /) -> Tensor:
    r"""Flattens element of general Hilbert space, skips over scalars."""
    return torch.cat([x.flatten() for x in iter_tensors(x)])


def get_shapes(x: Module | Tree, /) -> list[tuple[int, ...]]:
    r"""Return the shapes of the tensors."""
    return [item.shape for item in iter_tensors(x)]


def get_grads(x: Module | Tree, /) -> list[Tensor]:
    r"""Return a cloned detached copy of the gradients."""
    return [
        w.grad.clone().detach()
        for w in iter_tensors(x)
        if w.requires_grad and w.grad is not None
    ]


def get_norm(x: Nested[Tensor], /, *, normalize: bool = True) -> Tensor:
    r"""Compute the (normalized) 2-norm of a tensor."""
    flattened = flatten_nested_tensor(x)
    if normalize:
        return torch.sqrt(torch.mean(flattened**2))
    return torch.sqrt(torch.sum(flattened**2))


@overload
def make_tensors_parameters(x: Tensor, /) -> nn.Parameter: ...
@overload
def make_tensors_parameters(x: Scalar, /) -> Scalar: ...
@overload
def make_tensors_parameters[T](x: Mapping[str, T], /) -> dict[str, T]: ...
@overload
def make_tensors_parameters[T](x: Sequence[T], /) -> tuple[T, ...]: ...
def make_tensors_parameters(x, /):
    r"""Make tensors parameters."""
    # FIXME: https://github.com/python/cpython/issues/106246. Use match-case when fixed.
    if isinstance(x, Tensor):
        return nn.Parameter(x) if not isinstance(x, nn.Parameter) else x
    if isinstance(x, Scalar.__value__):
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
    r"""Test a forward pass."""
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
    r"""Test a backward pass."""
    try:
        r = get_norm(outputs)
        r.backward()
        gradients = get_grads(parameters)
        zero_grad(parameters)
    except Exception as exc:
        raise RuntimeError("Model failed backward pass!") from exc

    ref_grads = (
        deepcopy(gradients)
        if reference_gradients is None
        else list(iter_tensors(reference_gradients))
    )

    # check gradients
    assert_close(gradients, ref_grads)

    return gradients, ref_grads


@overload
def check_jit_scripting(module: Module, /) -> Module: ...
@overload
def check_jit_scripting(func: Func, /) -> Func: ...
def check_jit_scripting(arg: Func | Module, /) -> Func | Module:
    r"""Test JIT compilation."""
    try:
        scripted = jit.script(arg)
    except Exception as exc:
        raise RuntimeError("Model JIT compilation Failed!") from exc
    return scripted


@overload
def check_jit_serialization[M: Module](
    module: M, /, *, device: DeviceArg = ...
) -> M: ...
@overload
def check_jit_serialization(func: Func, /, *, device: DeviceArg = ...) -> Func: ...
def check_jit_serialization(
    arg: Func | Module, /, *, device: DeviceArg = None
) -> Func | Module:
    r"""Test saving and loading of JIT compiled model."""
    with tempfile.TemporaryFile() as file:
        try:
            jit.save(arg, file)
            file.seek(0)
        except Exception as exc:
            raise RuntimeError("Model saving failed!") from exc

        try:
            loaded = jit.load(file, map_location=device)
        except Exception as exc:
            raise RuntimeError("Model loading failed!") from exc
    return loaded


@overload
def check_jit(module: Module, /, *, device: DeviceArg = ...) -> Module: ...
@overload
def check_jit(func: Func, /, *, device: DeviceArg = ...) -> Func: ...
def check_jit(arg: Func | Module, /, *, device: DeviceArg = None) -> Func | Module:
    r"""Test JIT compilation+serialization."""
    # check if scripting and serialization works
    scripted = check_jit_scripting(arg)
    loaded = check_jit_serialization(scripted, device=device)
    return loaded


def check_initialization[M: Module](
    module_type: type[M],
    /,
    *,
    args: Sequence[Tree] = (),
    kwargs: Mapping[str, Tree] = EMPTY_MAP,
) -> M:
    r"""Test initialization of a module."""
    if not issubclass(module_type, Module):
        raise TypeError(f"Unsupported type {type(module_type)} for `obj`!")

    try:
        module = module_type(*args, **kwargs)
    except Exception as exc:
        raise RuntimeError("Model initialization failed!") from exc

    return module


def check_optim(
    model: Module,
    /,
    *,
    input_args: Sequence[Tree] = (),
    input_kwargs: Mapping[str, Tree] = EMPTY_MAP,
    niter: int = 4,
) -> None:
    r"""Check if model can be optimized."""
    with torch.no_grad():
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        original_outputs = model(*input_args, **input_kwargs)
        original_loss = get_norm(original_outputs)
        # original_params = [w.clone().detach() for w in model.parameters()]

    loss = original_loss

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


def check_model(
    model: Module,
    /,
    *,
    # input arguments
    input_args: Sequence[Tree] = (),
    input_kwargs: Mapping[str, Tree] = EMPTY_MAP,
    # reference arguments
    reference_gradients: Optional[Nested[Tensor]] = None,
    reference_model: Optional[Module] = None,
    reference_shapes: Optional[list[tuple[int, ...]]] = None,
    reference_outputs: Optional[Nested[Tensor]] = None,
    # extra arguments
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
    make_inputs_parameters: bool = True,
    test_jit: bool = True,
    test_optim: bool = False,
) -> None:
    r"""Check a module, function or model class."""
    # region get name and logger -------------------------------------------------------
    if not isinstance(model, Module):
        raise TypeError("Expected Module!")

    model_name = model.__class__.__name__
    logger = __logger__.getChild(model_name) if logger is None else logger
    # endregion get name and logger ----------------------------------------------------

    check_object(
        model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        reference_gradients=reference_gradients,
        reference_model=reference_model,
        reference_shapes=reference_shapes,
        reference_outputs=reference_outputs,
        device=device,
        logger=logger,
        make_inputs_parameters=make_inputs_parameters,
        test_jit=test_jit,
        test_optim=test_optim,
    )


def check_class(
    model_class: type[Module],
    /,
    *,
    init_args: Sequence[Any] = (),
    init_kwargs: Mapping[str, Any] = EMPTY_MAP,
    # input arguments
    input_args: Sequence[Tree] = (),
    input_kwargs: Mapping[str, Tree] = EMPTY_MAP,
    # reference arguments
    reference_gradients: Optional[Nested[Tensor]] = None,
    reference_model: Optional[Module] = None,
    reference_shapes: Optional[list[tuple[int, ...]]] = None,
    reference_outputs: Optional[Nested[Tensor]] = None,
    # extra arguments
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
    make_inputs_parameters: bool = True,
    test_jit: bool = True,
    test_optim: bool = False,
) -> None:
    r"""Test a model class."""
    # region get name and logger -------------------------------------------------------
    if not issubclass(model_class, Module):
        raise TypeError("Expected Module subclass!")

    class_name = model_class.__name__
    logger = logger if logger is not None else __logger__.getChild(class_name)
    # endregion get name and logger ----------------------------------------------------

    # region get initialized model if class --------------------------------------------
    model = check_initialization(model_class, args=init_args, kwargs=init_kwargs)
    logger.info(">>> Initialization ✔ ")
    # endregion get initialized model if class --------------------------------------------

    check_model(
        model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        reference_gradients=reference_gradients,
        reference_model=reference_model,
        reference_shapes=reference_shapes,
        reference_outputs=reference_outputs,
        device=device,
        logger=logger,
        make_inputs_parameters=make_inputs_parameters,
        test_jit=test_jit,
        test_optim=test_optim,
    )


def check_function(
    func: Func,
    /,
    *,
    # input arguments
    input_args: Sequence[Tree] = (),
    input_kwargs: Mapping[str, Tree] = EMPTY_MAP,
    # reference arguments
    reference_gradients: Optional[Nested[Tensor]] = None,
    reference_model: Optional[Module] = None,
    reference_shapes: Optional[list[tuple[int, ...]]] = None,
    reference_outputs: Optional[Nested[Tensor]] = None,
    # extra arguments
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
    make_inputs_parameters: bool = True,
    test_jit: bool = True,
    test_optim: bool = False,
) -> None:
    r"""Test a model class."""
    # region get name and logger -------------------------------------------------------
    if isinstance(func, Module):
        raise TypeError("For Modules, Use `check_model` instead!")

    if not callable(func):
        raise TypeError("Expected callable!")

    func_name = func.__name__
    logger = __logger__.getChild(func_name) if logger is None else logger
    # endregion get name and logger ----------------------------------------------------

    # test with initialized model
    check_object(
        func,
        input_args=input_args,
        input_kwargs=input_kwargs,
        reference_gradients=reference_gradients,
        reference_model=reference_model,
        reference_shapes=reference_shapes,
        reference_outputs=reference_outputs,
        device=device,
        logger=logger,
        make_inputs_parameters=make_inputs_parameters,
        test_jit=test_jit,
        test_optim=test_optim,
    )


def check_object(
    obj: type[Module] | Module | Func,
    *,
    init_args: Sequence[Any] = (),
    init_kwargs: Mapping[str, Any] = EMPTY_MAP,
    # input arguments
    input_args: Sequence[Tree] = (),
    input_kwargs: Mapping[str, Tree] = EMPTY_MAP,
    # reference arguments
    reference_gradients: Optional[Nested[Tensor]] = None,
    reference_model: Optional[Module] = None,
    reference_shapes: Optional[list[tuple[int, ...]]] = None,
    reference_outputs: Optional[Nested[Tensor]] = None,
    # extra arguments
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
    make_inputs_parameters: bool = True,
    test_jit: bool = True,
    test_optim: bool = False,
) -> None:
    r"""Check a module, function or model class."""
    # region get name and logger -------------------------------------------------------
    model: Module | Func
    match obj:
        case type() if issubclass(obj, Module):
            model_name = obj.__name__
        case Module() as model:
            model_name = model.__class__.__name__
        case Callable() as func:  # type: ignore[misc]
            model_name = func.__name__  # type: ignore[unreachable]
        case _:
            raise TypeError(f"Unsupported type {type(obj)} for `obj`!")

    # initialize logger
    logger = __logger__.getChild(model_name) if logger is None else logger
    # endregion get name and logger ----------------------------------------------------

    # region get initialized model if class --------------------------------------------
    match obj:
        case type() as cls:
            model = check_initialization(cls, args=init_args, kwargs=init_kwargs)
        case Module() as model:
            pass
        case Callable() as func:  # type: ignore[misc]
            model = func  # type: ignore[unreachable]
        case _:
            raise TypeError(f"Unsupported type {type(obj)} for `obj`!")

    logger.info(">>> Initialization ✔ ")
    # endregion get initialized model if class -----------------------------------------

    # region get parameters ------------------------------------------------------------
    model_parameters = get_parameters(model) if isinstance(model, Module) else []

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
        assert reference_outputs is None, "Both reference model & outputs given!"
        assert reference_gradients is None, "Both reference model & gradients given!"

        try:
            reference_model.to(device=device)
            reference_outputs = reference_model(*input_args, **input_kwargs)
            reference_parameters = get_parameters(reference_model) + input_parameters
            assert reference_outputs is not None
            r = get_norm(reference_outputs)
            r.backward()
            reference_gradients = get_grads(reference_parameters)
            zero_grad(reference_parameters)
        except Exception as exc:
            raise RuntimeError("Reference model failed forward/backward pass!") from exc
        logger.info(">>> Reference model forward/backward ✔ ")
    # endregion get reference model ----------------------------------------------------

    # region change device -------------------------------------------------------------
    if isinstance(model, Module):
        if device is None:
            device = get_device(model)

        try:  # cast model
            model = model.to(device=device)
        except Exception as exc:
            raise RuntimeError("Couldn't move model to device!") from exc

        # cast parameters
    try:  # cast other parameters
        input_args = to_device(input_args, device=device)
        if reference_outputs is not None:
            reference_outputs = to_device(reference_outputs, device=device)
        if reference_gradients is not None:
            reference_gradients = to_device(reference_gradients, device=device)
    except Exception as exc:
        raise RuntimeError("Couldn't move inputs to device!") from exc

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
        assert isinstance(model, Module), "Cannot test optimization of function!"
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
    check_object(
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
    check_object(
        loaded_model,
        input_args=input_args,
        reference_outputs=reference_outputs,
        reference_shapes=reference_shapes,
        reference_gradients=reference_gradients,
        logger=__logger__.getChild(f"{model_name}@DESERIALIZED"),
        test_jit=False,
    )
    # endregion check loaded forward/backward pass -------------------------------------
