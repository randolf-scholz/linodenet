r"""Blueprints for PyTorch models and tensors."""
# ruff: noqa: SIM103

__all__ = [
    # constants
    "MAX_SHAPE",
    # types
    "ModelBlueprint",
    "TensorBlueprint",
    # functions
    "infer_tensor_blueprint",
    "initialize_tensor",
    "is_model_blueprint",
    "is_tensor_blueprint",
    "_small_tensor_to_json",
    "validate_tensor_blueprint",
]

from typing import Any, NotRequired, ReadOnly, TypeGuard

from torch import Tensor, nn
from torch.export import ExportedProgram

from .core import (
    BLUEPRINT_REGISTRY,
    INFER_ARGS_REGISTRY,
    JSON,
    Args,
    Blueprint,
    Identifier,
    JSON_Value,
    _infer_object_blueprint,
    _initialize_object,
    _naive_serializer,
    infer_args,
    is_blueprint,
    is_object_blueprint,
)

MAX_SHAPE = (5, 5)
r"""CONF: Maximum tensor shape to inline as lists in hyperparameters."""


class ModelBlueprint[T: nn.Module = nn.Module](Blueprint[T]):
    r"""A blueprint that allows initializing a ``nn.Module``."""

    __module_name__: ReadOnly[str]
    __class_name__: ReadOnly[str]
    __module_version__: NotRequired[ReadOnly[str]]

    __args__: ReadOnly[list[object]]
    __kwargs__: ReadOnly[dict[Identifier, object]]
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


def _infer_model_blueprint[T: nn.Module](
    arg: T, /, *, verify_init: bool = True
) -> ModelBlueprint[T]:
    return _infer_object_blueprint(arg, verify_init=verify_init)


def _initialize_model_blueprint[T: nn.Module](spec: Blueprint[T], /) -> T:
    obj: T = _initialize_object(spec)
    assert isinstance(obj, nn.Module)
    return obj


def _small_tensor_to_json(value: Tensor, /) -> JSON_Value:
    assert isinstance(value, Tensor)
    if value.numel() == 1:
        return value.item()
    if value.ndim <= len(MAX_SHAPE) and value.shape <= MAX_SHAPE:
        return value.tolist()
    raise NotImplementedError(f"Tensor shape {tuple(value.shape)!r} exceeds MAX_SHAPE.")


def _model_blueprint_to_json[T: nn.Module](spec: Blueprint[T], /) -> JSON:
    # As an extra step, we convert small tensors in the args/kwargs to lists,
    # so that they do not need to be stored as separate objects.
    if not is_model_blueprint(spec):
        raise TypeError("Expected a model blueprint dictionary.")

    def _map_small_tensors(arg: object) -> JSON_Value:
        match arg:
            case Tensor():
                return _small_tensor_to_json(arg)
            case list() | tuple():
                return [_map_small_tensors(a) for a in arg]
            case dict(mapping) if not is_blueprint(mapping):
                return {k: _map_small_tensors(v) for k, v in mapping.items()}
            case other:
                return _naive_serializer(other)

    return {
        "__args__": _map_small_tensors(spec["__args__"]),
        "__kwargs__": _map_small_tensors(spec["__kwargs__"]),
        **{
            key: _naive_serializer(value)
            for key, value in spec.items()
            if key not in ("__args__", "__kwargs__")
        },
    }


BLUEPRINT_REGISTRY.register_initializer(is_model_blueprint, _initialize_model_blueprint)
BLUEPRINT_REGISTRY.register_serializer(is_model_blueprint, _model_blueprint_to_json)
BLUEPRINT_REGISTRY.register_maker(nn.Module, _infer_model_blueprint)


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


class TensorBlueprint[T: Tensor = Tensor](Blueprint[T]):
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


def infer_tensor_blueprint[T: Tensor](tensor: T, /) -> TensorBlueprint[T]:
    return {
        "__tensor__": tensor,
        "__dtype__": str(tensor.dtype),
        "__shape__": list(tensor.shape),
    }


def initialize_tensor[T: Tensor](spec: Blueprint[T], /) -> T:
    if not is_tensor_blueprint(spec):
        raise TypeError("Expected a tensor blueprint dictionary.")
    tensor = spec["__tensor__"]
    validate_tensor_blueprint(tensor, spec)
    return tensor


def validate_tensor_blueprint[T: Tensor](tensor: T, spec: Blueprint[T], /) -> None:
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


# Register nn.Module as a base class for blueprints
BLUEPRINT_REGISTRY.register_initializer(is_tensor_blueprint, initialize_tensor)
BLUEPRINT_REGISTRY.register_validator(is_tensor_blueprint, validate_tensor_blueprint)
BLUEPRINT_REGISTRY.register_maker(Tensor, infer_tensor_blueprint)
