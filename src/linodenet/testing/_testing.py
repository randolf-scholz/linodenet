r"""Utility functions for testing.

Naming convention:

- `assert_*`: should return `None` if the assertion passes.
- `check_*`: should return the output of the function.
- `is_*`, `all_*`, `any_*`: should return a boolean / TypeIs / TypeGuard.
"""

__all__ = [
    # check functions
    "check_class",
    "check_function",
    "check_model",
    "check_object",
    # helper functions
    "all_finite",
    "assert_close",
    "assert_is_trainable",
    "assert_jit_compatible",
    "assert_signatures_compatible",
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


def assert_signatures_compatible(func: Callable, reference: Callable) -> None:
    r"""Assert that functions signature is wider than reference."""
    fun_sig = inspect.signature(func)
    ref_sig = inspect.signature(reference)

    for param in ref_sig.parameters:
        if param not in fun_sig.parameters:
            raise AssertionError(f"Parameter {param} not in function signature!")
        ref_kind = ref_sig.parameters[param].kind
        param_kind = fun_sig.parameters[param].kind
        if param_kind != ref_kind:
            raise AssertionError(
                f"Parameter {param} has different kind! (expected {ref_kind}, got {param_kind})"
            )


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
            if not torch.allclose(tensor, reference, rtol=rtol, atol=atol):
                raise AssertionError
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
            raise TypeError(f"Unsupported type {type(values)}!")


def all_finite(x: Nested[Tensor], /) -> bool:
    r"""Check if all elements are finite."""
    match x:
        case Tensor() as tensor:
            return bool(torch.all(torch.isfinite(tensor)))
        case Mapping() as mapping:
            return all(all_finite(val) for val in mapping.values())
        case Sequence() as sequence:
            return all(all_finite(item) for item in sequence)
        case _:
            raise TypeError(f"Unsupported type {type(x)}!")


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
def to_device[T: Tensor](x: T, /, *, device: DeviceArg = ...) -> T: ...
@overload
def to_device[S: Scalar](x: S, /, *, device: DeviceArg = ...) -> S: ...
@overload
def to_device[*Ts](tup: tuple[*Ts], /, *, device: DeviceArg = ...) -> tuple[*Ts]: ...
@overload
def to_device[T](x: Sequence[T], /, *, device: DeviceArg = ...) -> list[T]: ...
@overload
def to_device[T](x: Mapping[str, T], /, *, device: DeviceArg = ...) -> dict[str, T]: ...
@overload
def to_device[T](x: T, /, *, device: DeviceArg = ...) -> T: ...
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
        case tuple(tup):
            return tuple(to_device(item, device=device) for item in tup)
        case Sequence() as seq:
            return [to_device(item, device=device) for item in seq]
        case Mapping() as mapping:
            return {key: to_device(val, device=device) for key, val in mapping.items()}
        case _:
            return x


def iter_tensors(x: Module | Tree, /) -> Iterator[Tensor]:
    r"""Yields the tensors of a model."""
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
    r"""Yields the parameters of the model."""
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
def make_tensors_parameters[S: Scalar](x: S, /) -> S: ...
@overload
def make_tensors_parameters[T](x: Mapping[str, T], /) -> dict[str, T]: ...
@overload
def make_tensors_parameters[*Ts](x: tuple[*Ts], /) -> tuple[*Ts]: ...
@overload
def make_tensors_parameters[T](x: Sequence[T], /) -> list[T]: ...
def make_tensors_parameters(x: Any, /) -> Any:
    r"""Make tensors parameters."""
    # FIXME: https://github.com/python/cpython/issues/106246. Use match-case when fixed.
    match x:
        case Tensor() as tensor:
            return nn.Parameter(x) if not isinstance(x, nn.Parameter) else x
        case scalar if isinstance(scalar, Scalar.__value__):
            return scalar
        case Mapping() as mapping:
            return {key: make_tensors_parameters(val) for key, val in mapping.items()}
        case tuple(tup):
            return tuple(make_tensors_parameters(item) for item in tup)
        case Sequence() as seq:
            return [make_tensors_parameters(item) for item in seq]
        case _:
            raise TypeError(f"Unsupported input type {type(x)!r}")


# endregion utility functions  for outputs (always tensor) -----------------------------


# region check helper functions --------------------------------------------------------


def _check_initialization[M: Module](
    module_type: type[M],
    /,
    *,
    init_args: Sequence[Tree] = (),
    init_kwargs: Mapping[str, Tree] = EMPTY_MAP,
) -> M:
    r"""Test initialization of a module."""
    if not isinstance(module_type, type):
        raise TypeError(f"Expected type, got {type(module_type)}!")

    if not issubclass(module_type, Module):
        raise TypeError(f"Unsupported type {type(module_type)} for `obj`!")

    try:
        module = module_type(*init_args, **init_kwargs)
    except Exception as exc:
        raise RuntimeError("Model initialization failed!") from exc

    return module


def _check_forward(
    func: Func,
    /,
    *,
    input_args: Sequence[Tree],
    input_kwargs: Mapping[str, Tree],
    # optional: reference outputs and shapes
    reference_values: Optional[Nested[Tensor]] = None,
    reference_shapes: Optional[list[tuple[int, ...]]] = None,
) -> Nested[Tensor]:
    r"""Test a forward pass."""
    try:
        outputs = func(*input_args, **input_kwargs)
    except Exception as exc:
        raise RuntimeError("Forward pass failed!!") from exc

    # validate shapes
    shapes = get_shapes(outputs)
    if reference_shapes is not None and shapes != reference_shapes:
        raise AssertionError(f"Shapes mismatch! {reference_shapes=} {shapes=}")

    # validate values
    if reference_values is not None:
        assert_close(outputs, reference_values)

    return outputs


def _check_backward(
    module_or_func: Module | Func,
    /,
    *,
    input_args: Sequence[Tree],
    input_kwargs: Mapping[str, Tree],
    # Optional: reference gradients
    reference_values: Optional[Nested[Tensor]] = None,
    reference_shapes: Optional[list[tuple[int, ...]]] = None,
    treat_inputs_as_parameters: bool = True,
) -> list[Tensor]:
    r"""Test a backward pass."""
    params: list[Tensor] = (
        get_parameters(module_or_func) if isinstance(module_or_func, Module) else []
    )

    if treat_inputs_as_parameters:
        input_args = make_tensors_parameters(input_args)
        input_kwargs = make_tensors_parameters(input_kwargs)
        params.extend(get_parameters(input_args))
        params.extend(get_parameters(input_kwargs))

    with torch.enable_grad():
        outputs = _check_forward(
            module_or_func,
            input_args=input_args,
            input_kwargs=input_kwargs,
        )

        # compute a simple scalar value.
        r = get_norm(outputs)

        try:
            r.backward()
        except Exception as exc:
            raise AssertionError("Model failed backward pass!") from exc

        # extract gradients and their shapes
        gradients = get_grads(params)
        shapes = get_shapes(gradients)

    # validate shapes
    if reference_shapes is not None and shapes != reference_shapes:
        raise AssertionError(f"Shapes mismatch! {reference_shapes=} {shapes=}")

    # validate values
    if reference_values is not None:
        assert_close(gradients, reference_values)

    return gradients


def _checked_jit_scriptable[M: Module | Func](arg: M, /) -> M:
    r"""Test JIT compilation."""
    try:
        scripted = jit.script(arg)
    except Exception as exc:
        raise AssertionError("Model JIT compilation Failed!") from exc
    return scripted


def _check_jit_serializable[M: Module | Func](
    arg: M, /, *, device: DeviceArg = "cpu"
) -> M:
    r"""Test saving and loading of JIT compiled model."""
    with tempfile.TemporaryFile() as file:
        try:
            jit.save(arg, file)
            file.seek(0)
        except Exception as exc:
            raise AssertionError("Model saving failed!") from exc

        try:
            loaded = jit.load(file, map_location=device)
        except Exception as exc:
            raise AssertionError("Model loading failed!") from exc
    return loaded


# endregion check helper functions -----------------------------------------------------


def assert_is_trainable(
    module: Module,
    /,
    *,
    input_args: Sequence[Tree] = (),
    input_kwargs: Mapping[str, Tree] = EMPTY_MAP,
    niter: int = 4,
    use_copy: bool = True,
) -> None:
    r"""Check if model can be optimized."""
    if not any(p.requires_grad for p in module.parameters()):
        raise AssertionError("No trainable parameters!")

    if use_copy:
        with torch.no_grad():
            model = deepcopy(module)
            input_args = deepcopy(input_args)
            input_kwargs = deepcopy(input_kwargs)
        # fix the gradient state
        for w, p in zip(model.parameters(), module.parameters(), strict=True):
            w.requires_grad_(p.requires_grad)
    else:
        model = module

    if not any(p.requires_grad for p in model.parameters()):
        raise AssertionError("No trainable parameters!")

    with torch.no_grad():
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        original_outputs = model(*input_args, **input_kwargs)
        original_loss = get_norm(original_outputs)
        # original_params = [w.clone().detach() for w in model.parameters()]

    loss = original_loss
    history: list[float] = [float(original_loss.item())]

    # perform iterations
    for _ in range(niter):
        model.zero_grad(set_to_none=True)
        outputs = model(*input_args, **input_kwargs)
        loss = get_norm(outputs)
        assert loss.isfinite()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

    # check that the loss decreased
    if loss >= original_loss:
        raise AssertionError(f"Loss did not decrease! \n{history=}")


def assert_jit_compatible[M: Module | Func](
    module_or_function: M,
    /,
    *,
    input_args: Sequence[Tree],
    input_kwargs: Mapping[str, Tree],
    # optional arguments
    check_is_trainable: bool = True,
    device: DeviceArg = "cpu",
) -> None:
    r"""Test if a model is compatible with JIT."""
    # get reference values from forward/backward pass
    ref_outs = _check_forward(
        module_or_function,
        input_args=input_args,
        input_kwargs=input_kwargs,
    )

    ref_grads = _check_backward(
        module_or_function,
        input_args=input_args,
        input_kwargs=input_kwargs,
    )

    # script the module
    scripted_obj = _checked_jit_scriptable(module_or_function)
    # perform forward pass
    scripted_outs = _check_forward(
        scripted_obj,
        input_args=input_args,
        input_kwargs=input_kwargs,
        reference_values=ref_outs,
        reference_shapes=get_shapes(ref_outs),
    )
    # perform backward pass
    _check_backward(
        scripted_obj,
        input_args=input_args,
        input_kwargs=input_kwargs,
        reference_values=ref_grads,
        reference_shapes=get_shapes(ref_grads),
    )
    if check_is_trainable and isinstance(scripted_obj, Module):
        assert_is_trainable(
            scripted_obj,
            input_args=input_args,
            input_kwargs=input_kwargs,
        )

    # check serialization
    deserialized_obj = _check_jit_serializable(scripted_obj, device=device)

    # perform forward pass
    deserialized_outs = _check_forward(
        deserialized_obj,
        input_args=input_args,
        input_kwargs=input_kwargs,
        reference_values=ref_outs,
        reference_shapes=get_shapes(ref_outs),
    )
    # perform backward pass
    _check_backward(
        deserialized_obj,
        input_args=input_args,
        input_kwargs=input_kwargs,
        reference_values=ref_grads,
        reference_shapes=get_shapes(ref_grads),
    )

    if check_is_trainable and isinstance(deserialized_obj, Module):
        assert_is_trainable(
            deserialized_obj,
            input_args=input_args,
            input_kwargs=input_kwargs,
        )


def check_model(
    model_or_func: Module | Func,
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

    # # region get parameters ------------------------------------------------------------
    # model_parameters = get_parameters(model) if isinstance(model, Module) else []
    #
    # # get parameters of input tensors
    # if make_inputs_parameters:
    #     input_args = make_tensors_parameters(input_args)
    #     input_kwargs = make_tensors_parameters(input_kwargs)
    #     input_parameters = get_parameters((input_args, input_kwargs))
    # else:
    #     input_parameters = []
    #
    # parameters = model_parameters + input_parameters
    # # endregion get parameters model ---------------------------------------------------
    #
    # # region get reference model -------------------------------------------------------
    # if reference_model is not None:
    #     assert reference_outputs is None, "Both reference model & outputs given!"
    #     assert reference_gradients is None, "Both reference model & gradients given!"
    #
    #     try:
    #         reference_model.to(device=device)
    #         reference_outputs = reference_model(*input_args, **input_kwargs)
    #         reference_parameters = get_parameters(reference_model) + input_parameters
    #         assert reference_outputs is not None
    #         r = get_norm(reference_outputs)
    #         r.backward()
    #         reference_gradients = get_grads(reference_parameters)
    #         zero_grad(reference_parameters)
    #     except Exception as exc:
    #         raise RuntimeError("Reference model failed forward/backward pass!") from exc
    #     logger.info(">>> Reference model forward/backward ✔ ")
    #     # endregion get reference model ----------------------------------------------------

    # region change device -------------------------------------------------------------
    model, input_args, input_kwargs = to_device(
        (model, input_args, input_kwargs), device=device
    )
    # endregion change device ----------------------------------------------------------

    # region check forward pass --------------------------------------------------------
    outputs = _check_forward(
        model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        reference_values=reference_outputs,
        reference_shapes=reference_shapes,
    )
    logger.info(">>> Forward ✔ ")
    # endregion check forward pass -----------------------------------------------------

    # region check backward pass -------------------------------------------------------
    gradients = _check_backward(
        model,
        outputs=outputs,
        reference_values=reference_gradients,
    )
    assert all_finite(gradients), "Gradients are not finite!"
    logger.info(">>> Backward ✔ ")
    # endregion check backward pass ----------------------------------------------------

    if test_optim:
        assert_is_trainable(model, input_args=input_args, input_kwargs=input_kwargs)

    if test_jit:
        assert_jit_compatible(model, input_args=input_args, input_kwargs=input_kwargs)


def check_class(
    model_class: type[Module],
    /,
    *,
    init_args: Sequence[Any] = (),
    init_kwargs: Mapping[str, Any] = EMPTY_MAP,
    # input arguments
    **check_model_kwargs: Any,
) -> None:
    r"""Test a model class."""
    model = _check_initialization(
        model_class, init_args=init_args, init_kwargs=init_kwargs
    )
    check_model(model, **check_model_kwargs)
