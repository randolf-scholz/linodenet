r"""Blueprints for PyTorch models and tensors."""
# ruff: noqa: SIM103

__all__ = [
    "ModelBlueprint",
    "is_model_blueprint",
    "TensorBlueprint",
    "is_tensor_blueprint",
    "infer_tensor_blueprint",
    "validate_tensor_blueprint",
    "initialize_tensor",
    "initialize_from_dict",
]

from importlib import import_module
from typing import Any, NotRequired, ReadOnly, TypedDict, TypeGuard, cast

from torch import Tensor, nn
from torch.export import ExportedProgram

from blueprint.config import SupportsFromConfig
from blueprint.core import (
    BLUEPRINT_REGISTRY,
    INFER_ARGS_REGISTRY,
    JSON,
    Args,
    ArgValue,
    Blueprint,
    Identifier,
    Makes,
    _infer_object_blueprint,
    infer_args,
    initialize_object,
    is_blueprint,
    is_object_blueprint,
)


class ModelBlueprint[T: nn.Module = nn.Module](TypedDict):
    r"""A blueprint that allows initializing a ``nn.Module``."""

    __module_name__: ReadOnly[str]
    __class_name__: ReadOnly[str]
    __module_version__: NotRequired[ReadOnly[str]]

    __args__: ReadOnly[list[ArgValue]]
    __kwargs__: ReadOnly[dict[Identifier, ArgValue]]
    # **DunderKey: object (reserved for future use)


def is_model_blueprint(arg: object, /) -> TypeGuard[ModelBlueprint]:
    r"""Check if the argument is a valid model blueprint.

    Note: This will import the module and check if the class is a subclass of nn.Module.
    """
    if not is_object_blueprint(arg):
        return False
    # import the class and check if it is a subclass of nn.Module
    module_name = arg["__module_name__"]
    class_name = arg["__class_name__"]
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
    except ImportError, AttributeError:
        return False
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        return False
    return True


def _infer_nn_linear(model: nn.Linear, /) -> Args:
    return [], {
        "in_features": model.in_features,
        "out_features": model.out_features,
        "bias": model.bias is not None,
    }


def _infer_nn_sequential(model: nn.Sequential, /) -> Args:
    return list(model), {}


def _infer_nn_modulelist(model: nn.ModuleList, /) -> Args:
    return [list(model)], {}


def _infer_nn_moduledict(model: nn.ModuleDict, /) -> Args:
    return [dict(model)], {}


def _infer_exported_module(model: ExportedProgram, /) -> Args:
    return infer_args(model.module())


# Register special inference functions for some common nn.Modules
INFER_ARGS_REGISTRY.register(nn.Sequential, _infer_nn_sequential)
INFER_ARGS_REGISTRY.register(nn.ModuleList, _infer_nn_modulelist)
INFER_ARGS_REGISTRY.register(nn.ModuleDict, _infer_nn_moduledict)
INFER_ARGS_REGISTRY.register(nn.Linear, _infer_nn_linear)
INFER_ARGS_REGISTRY.register(ExportedProgram, _infer_exported_module)


class TensorBlueprint[T: Tensor = Tensor](TypedDict):
    r"""A pseudo-blueprint that wraps a tensor value."""

    __tensor__: ReadOnly[Any]
    __dtype__: ReadOnly[str]
    __shape__: ReadOnly[list[int]]


def is_tensor_blueprint(arg: object, /) -> TypeGuard[TensorBlueprint]:
    if not is_blueprint(arg):
        return False
    if not TensorBlueprint.__required_keys__.issubset(arg.keys()):
        return False
    return True


def infer_tensor_blueprint(tensor: Tensor) -> TensorBlueprint:
    return {
        "__tensor__": tensor,
        "__dtype__": str(tensor.dtype),
        "__shape__": list(tensor.shape),
    }


def validate_tensor_blueprint(tensor: Tensor, spec: TensorBlueprint, /) -> None:
    if not is_tensor_blueprint(spec):
        raise TypeError("Invalid tensor spec.")
    expected_dtype = spec["__dtype__"]
    expected_shape = spec["__shape__"]
    if str(tensor.dtype) != expected_dtype:
        raise ValueError(
            f"Tensor dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
        )
    if list(tensor.shape) != list(expected_shape):
        raise ValueError(
            f"Tensor shape mismatch: expected {expected_shape}, got {list(tensor.shape)}"
        )


def initialize_tensor(spec: TensorBlueprint, /) -> Tensor:
    if not is_tensor_blueprint(spec):
        raise TypeError("Expected a tensor blueprint dictionary.")
    tensor = spec["__tensor__"]
    validate_tensor_blueprint(tensor, spec)
    return tensor


MAX_SHAPE = (5, 5)
r"""Maximum tensor shape to inline as lists in hyperparameters."""


def tensor_to_json(spec: TensorBlueprint, /) -> JSON:
    value = spec["__tensor__"]
    if value.numel() == 1:
        return _value_to_json(value.item())
    if value.ndim <= len(MAX_SHAPE) and value.shape <= MAX_SHAPE:
        return _value_to_json(value.tolist())
    raise NotImplementedError(f"Tensor shape {tuple(value.shape)!r} exceeds MAX_SHAPE.")


# Register nn.Module as a base class for blueprints
BLUEPRINT_REGISTRY.register_initializer(is_tensor_blueprint, initialize_tensor)
BLUEPRINT_REGISTRY.register_validator(is_tensor_blueprint, validate_tensor_blueprint)
BLUEPRINT_REGISTRY.register_maker(Tensor, infer_tensor_blueprint)


def _infer_model_blueprint[T: nn.Module](
    arg: T, /, *, verify_init: bool = True
) -> ModelBlueprint[T]:
    return _infer_object_blueprint(arg, verify_init=verify_init)


def _initialize_model_blueprint[T: nn.Module](spec: Blueprint[T], /) -> T:
    obj: T = initialize_object(spec)
    assert isinstance(obj, nn.Module)
    return obj


BLUEPRINT_REGISTRY.register_initializer(is_model_blueprint, _initialize_model_blueprint)
BLUEPRINT_REGISTRY.register_maker(nn.Module, _infer_model_blueprint)


def initialize_from_dict[M: nn.Module](cfg: Makes[M], /) -> M:
    r"""Initialize a module from a dictionary.

    Args:
        cfg: A dictionary containing the default configuration of the class.

    Note:
        The configuration must provide the keys `__module__` and `__name__`.
        The function will attempt to import the module and class and initialize it.
    """
    config = dict(cfg)

    if (lib_name := config.pop("__module__", None)) is None:
        raise ValueError(f"Expected {config=} to contain '__module__'")
    if (cls_name := config.pop("__name__", None)) is None:
        raise ValueError(f"Expected {config=} to contain '__name__'")

    try:  # import the module
        library = import_module(lib_name)
    except ModuleNotFoundError as exc:
        exc.add_note(f"Failed to import {lib_name=}")
        raise

    try:  # import the class from the module
        cls = getattr(library, cls_name)
    except AttributeError as exc:
        exc.add_note(f"Failed to import {cls_name} from {lib_name}")
        raise
    if not issubclass(cls, nn.Module):
        raise TypeError(f"Expected a subclass of {nn.Module}, but got {cls}")

    # attempt to initialize the class
    #  check if classmethod from_config is available
    if issubclass(cls, SupportsFromConfig):
        try:
            module = cls.from_config(config)
        except Exception as exc:
            exc.add_note(f"Failed to initialize {cls} with {config=}")
            raise
        return module

    try:
        module = cls(**config)
    except Exception as exc:
        exc.add_note(f"Failed to initialize {cls} with {config=}")
        raise
    return module
