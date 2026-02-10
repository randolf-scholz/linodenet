r"""Configuration protocols and utilities."""

# ruff: noqa: SIM103
__all__ = [
    # types
    "JSON",
    # constants
    "BLUEPRINT_REGISTRY",
    "INFER_ARGS_REGISTRY",
    # blueprint types
    "BluePrint",
    "InferArgsRegistry",
    "ObjectBluePrint",
    "ModelBluePrint",
    "TensorBluePrint",
    "BluePrintRegistry",
    "is_blueprint",
    "is_model_blueprint",
    "blueprint_to_json",
    # protocols
    "SupportsConfig",
    "has_config",
    "is_config",
    "SupportsDefaultConfig",
    "has_default_config",
    "SupportsFromConfig",
    # functions
    "infer_args",
    "initialize",
    "initialize_from_args",
    "infer_blueprint",
    "validate_blueprint",
]


import inspect
import os
from collections.abc import Callable, Iterator, Mapping
from inspect import Parameter
from typing import (
    Any,
    ClassVar,
    NotRequired,
    Protocol,
    ReadOnly,
    Self,
    TypedDict,
    TypeGuard,
    cast,
    runtime_checkable,
)

from torch import Tensor, nn
from torch.export import ExportedProgram

MAX_SHAPE = (5, 5)
r"""Maximum tensor shape to inline as lists in hyperparameters."""


# region config protocols --------------------------------------------------------------
class SupportsConfig(Protocol):
    r"""Models that support a hyperparameter dictionary.

    A hyperparameter dictionary should be such that

    type(model)(**model.config)

    recovers the model.
    """

    @property
    def config(self) -> dict[ArgKey, ArgValue]: ...


def is_config(arg: object, /) -> TypeGuard[dict[ArgKey, ArgValue]]:
    if not isinstance(arg, dict):
        return False
    return all(is_arg_key(key) and is_arg_value(value) for key, value in arg.items())


def has_config(arg: object, /) -> TypeGuard[SupportsConfig]:
    if (config := getattr(arg, "config", None)) is None:
        return False
    return is_config(config)


class SupportsDefaultConfig(Protocol):
    r"""Models that have a default configuration dataclass."""

    DEFAULT_CONFIG: ClassVar[type]


def has_default_config(arg: object, /) -> TypeGuard[SupportsDefaultConfig]:
    if not isinstance(arg, type):
        arg = type(arg)
    default_config = getattr(arg, "DEFAULT_CONFIG", None)
    return default_config is not None


@runtime_checkable
class SupportsFromConfig(Protocol):
    r"""Models that can be explicitly initialized from a configuration dictionary."""

    @classmethod
    def from_config(cls, config: dict[ArgKey, ArgValue], /) -> Self: ...


# endregion config protocols -----------------------------------------------------------


type Makes[T] = type[T] | BluePrint[T]  # dataclass
type FilePath = str | os.PathLike[str]

# region key and value types -----------------------------------------------------------
type Key = str  # any string.
type ArgKey = str  # identifier
type DunderKey = str  # identifier & dunder
type SunderKey = str  # identifier & sunder (not dunder)
type PrivateKey = str  # identifier & starts with _ (includes dunder and sunder)


# The types allowed in the config dictionary.
# configs should not contain ModelBluePrints or TensorSpecs directly, but actual models and tensors
type ArgLeaf = None | bool | int | float | str | Tensor | nn.Module
type ArgValue = ArgLeaf | list[ArgValue] | tuple[ArgValue, ...] | dict[str, ArgValue]
type POArgs = list[ArgValue]
type KWArgs = dict[ArgKey, ArgValue]
type Args = tuple[POArgs, KWArgs]

type JSON_Leaf = None | bool | int | float | str
type JSON_Value = JSON_Leaf | list[JSON_Value] | dict[str, JSON_Value]
type JSON = dict[str, JSON_Value]

# FIXME: do we need an intermediate Spec type for in memory spec, where only
# models are replaces with ModelBluePrint?


def is_dunder_key(value: object, /) -> TypeGuard[DunderKey]:
    return (
        isinstance(value, str)
        and len(value) > 4
        and value.isidentifier()
        and value.startswith("__")
        and value.endswith("__")
        and not value.startswith("___")
        and not value.endswith("___")
    )


def is_sunder_key(value: object, /) -> TypeGuard[SunderKey]:
    return (
        isinstance(value, str)
        and len(value) > 2
        and value.isidentifier()
        and value.startswith("_")
        and value.endswith("_")
        and not value.startswith("__")
        and not value.endswith("__")
    )


def is_private_key(value: object, /) -> TypeGuard[PrivateKey]:
    return isinstance(value, str) and value.isidentifier() and value.startswith("_")


def is_arg_key(value: object, /) -> TypeGuard[ArgKey]:
    return isinstance(value, str) and value.isidentifier()


def is_arg_value(arg: object, /) -> TypeGuard[ArgValue]:
    match arg:
        case None | bool() | int() | float() | str() | Tensor() | nn.Module():
            return True
        case list() | tuple():
            return all(is_arg_value(value) for value in arg)
        case dict():
            return all(is_arg_value(value) for value in arg.values())
        case _:
            return False


def is_blueprint_key(value: object, /) -> TypeGuard[DunderKey | SunderKey]:
    return (
        isinstance(value, str)
        and value.isidentifier()
        and (value.startswith("_") and value.endswith("_"))
    )


# endregion allowed types --------------------------------------------------------------


# region blueprint types ---------------------------------------------------------------
class BluePrint[T](TypedDict):
    r"""A dictionary that can be used to initialize an object of type T.

    All keys must be dunder or sunder.
    """

    # dunder and sunder keys only (not expressible in the type system)


class ObjectBluePrint[T](TypedDict):
    r"""A dictionary that allows initializing an object."""

    __module_name__: ReadOnly[str]
    __class_name__: ReadOnly[str]
    __module_version__: NotRequired[ReadOnly[str]]

    __args__: ReadOnly[list[ArgValue]]
    __kwargs__: ReadOnly[dict[ArgKey, ArgValue]]
    # **DunderKey: object (reserved for future use)


class ModelBluePrint[T: nn.Module](TypedDict):
    r"""A blueprint that allows initializing a ``nn.Module``."""

    __module_name__: ReadOnly[str]
    __class_name__: ReadOnly[str]
    __module_version__: NotRequired[ReadOnly[str]]

    __args__: ReadOnly[list[ArgValue]]
    __kwargs__: ReadOnly[dict[ArgKey, ArgValue]]
    # **DunderKey: object (reserved for future use)


class TensorBluePrint[T: Tensor = Tensor](TypedDict):
    r"""A pseudo-blueprint that wraps a tensor value."""

    __tensor__: ReadOnly[Any]
    __dtype__: ReadOnly[str]
    __shape__: ReadOnly[list[int]]


def is_blueprint(arg: object, /) -> TypeGuard[BluePrint]:
    if not isinstance(arg, dict):
        return False
    if not all(is_blueprint_key(key) for key in arg):
        return False
    return True


def is_tensor_blueprint(arg: object, /) -> TypeGuard[TensorBluePrint]:
    if not is_blueprint(arg):
        return False
    if not TensorBluePrint.__required_keys__.issubset(arg.keys()):
        return False
    return True


def is_object_blueprint(arg: object, /) -> TypeGuard[ObjectBluePrint]:
    if not is_blueprint(arg):
        return False
    if not ObjectBluePrint.__required_keys__.issubset(arg.keys()):
        return False
    if not isinstance(arg.get("__module_name__"), str):
        return False
    if not isinstance(arg.get("__class_name__"), str):
        return False
    if not (
        isinstance(args := arg.get("__args__"), list)
        and all(is_arg_value(item) for item in args)
    ):
        return False
    if not (
        isinstance(kwargs := arg.get("__kwargs__"), dict)
        and all(
            is_arg_key(key) and is_arg_value(value) for key, value in kwargs.items()
        )
    ):
        return False
    return True


def is_model_blueprint(arg: object, /) -> TypeGuard[ModelBluePrint]:
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


# endregion blueprint types ------------------------------------------------------------


# region for inferring config from instance --------------------------------------------
def _infer_args_from_init(arg: object, /) -> Args:
    r"""Infers `*args` and `**kwargs` by matching model's attributes with `__init__`.

    This only works for models that are essentially dataclasses.
    """
    signature = inspect.signature(arg.__init__)  # type: ignore[misc]
    params = list(signature.parameters.values())

    if len(params) == 0:
        raise ValueError("illegal signature (no self parameter)")

    params = params[1:]  # skip 'self'

    if any(p.kind is Parameter.POSITIONAL_ONLY for p in params):
        raise ValueError("Positional-only parameters are not supported")

    if any(p.kind is Parameter.VAR_POSITIONAL for p in params):
        raise ValueError("*args parameters are not supported")

    if any(p.kind is Parameter.VAR_KEYWORD for p in params):
        raise ValueError("**kwargs parameters are not supported")

    # skip 'self'
    kwargs = {p.name: p for p in params}

    # validate names
    if not all(name.isidentifier() for name in kwargs):
        raise ValueError("All parameter names must be valid identifiers.")
    if any(is_dunder_key(name) for name in kwargs):
        raise ValueError("Parameter names cannot be dunder names.")

    if missing := {
        name
        for name in kwargs
        if not hasattr(arg, name) and kwargs[name].default is Parameter.empty
    }:
        raise ValueError(f"Missing model attributes for config: {sorted(missing)}")

    return [], {name: getattr(arg, name) for name in (kwargs.keys() - missing)}


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


class InferArgsRegistry(Mapping[type, Callable[[Any], Args]]):
    r"""Registry for functions that infer model args and kwargs from instances."""

    def __init__(self) -> None:
        self._registry: dict[type, Callable[[Any], Args]] = {}
        self.register(nn.Sequential, _infer_nn_sequential)
        self.register(nn.ModuleList, _infer_nn_modulelist)
        self.register(nn.ModuleDict, _infer_nn_moduledict)
        self.register(nn.Linear, _infer_nn_linear)
        self.register(ExportedProgram, _infer_exported_module)

    def __getitem__[T](self, key: type[T], /) -> Callable[[T], Args]:
        return self._registry[key]

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self) -> Iterator[type]:
        return iter(self._registry)

    def register[T](self, cls: type[T], fn: Callable[[T], Args], /) -> None:
        if cls in self._registry:
            raise ValueError(f"Model class {cls.__qualname__} is already registered.")
        self._registry[cls] = fn


INFER_ARGS_REGISTRY = InferArgsRegistry()


def infer_args(arg: object, /, *, verify_init: bool = True) -> Args:
    r"""Infer the configuration dictionary for a model.

    Args:
        arg: The model to infer the configuration from.
        verify_init: Whether to verify that the inferred config can be used
            to initialize the model.
    """
    args: POArgs
    kwargs: KWArgs
    if has_config(arg):
        args, kwargs = [], arg.config
    elif type(arg) in INFER_ARGS_REGISTRY:
        infer_fn = INFER_ARGS_REGISTRY[type(arg)]
        args, kwargs = infer_fn(arg)
    else:
        args, kwargs = _infer_args_from_init(arg)

    if verify_init:
        cls = type(arg)
        try:
            m = initialize_from_args(cls, args, kwargs)
        except Exception as exc:
            raise ValueError(
                f"Failed to verify inferred config for {cls.__qualname__}."
            ) from exc
        assert type(m) is cls

    assert is_config(kwargs)
    return args, kwargs


def initialize_from_args[T](cls: type[T], args: POArgs, kwargs: KWArgs, /) -> T:
    r"""Initialize a model from args and kwargs."""
    if not args and issubclass(cls, SupportsFromConfig):
        try:
            return cls.from_config(kwargs)
        except Exception as exc:
            raise ValueError(
                f"Failed to initialize {cls.__qualname__} from config."
            ) from exc

    try:
        return cls(*args, **kwargs)
    except Exception as exc:
        raise ValueError(
            f"Failed to initialize {cls.__qualname__} from args and kwargs."
        ) from exc


# endregion for inferring config from instance -----------------------------------------


def resolve_value(arg: ArgValue, /) -> Any:
    match arg:
        case blueprint if is_blueprint(blueprint):
            return initialize(blueprint)
        case list():
            return [resolve_value(item) for item in arg]
        case tuple():
            return tuple(resolve_value(item) for item in arg)
        case dict():
            return {key: resolve_value(item) for key, item in arg.items()}
        case _:
            return arg


def _initialize_object_blueprint[T](spec: BluePrint[T], /) -> T:
    if not is_object_blueprint(spec):
        raise TypeError("Expected an object blueprint dictionary.")

    module_name = spec["__module_name__"]
    class_name = spec["__class_name__"]

    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    args = [resolve_value(item) for item in spec["__args__"]]
    kwargs = {key: resolve_value(item) for key, item in spec["__kwargs__"].items()}

    return initialize_from_args(cls, args, kwargs)


def _initialize_tensor_blueprint[T: Tensor](spec: BluePrint[T], /) -> T:
    if not is_tensor_blueprint(spec):
        raise TypeError("Expected a tensor blueprint dictionary.")
    tensor = spec["__tensor__"]
    _validate_tensor_blueprint(tensor, spec)
    return cast("T", tensor)


def _validate_object_blueprint[T](arg: T | type[T], spec: BluePrint, /) -> None:
    if not is_object_blueprint(spec):
        raise TypeError("Invalid model spec.")
    if any(not key.isidentifier() for key in spec):
        raise TypeError("Model spec keys must be identifiers.")

    cls = arg if isinstance(arg, type) else arg.__class__
    expected_module = spec["__module_name__"]
    expected_name = spec["__class_name__"]

    if cls.__module__ != expected_module:
        raise ValueError(
            f"Module path mismatch: expected {expected_module}, got {cls.__module__}"
        )
    if cls.__qualname__ != expected_name:
        raise ValueError(
            f"Module name mismatch: expected {expected_name}, got {cls.__qualname__}"
        )


def _validate_tensor_blueprint(tensor: Tensor, spec: BluePrint, /) -> None:
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


def _infer_object_blueprint[T](
    arg: T, /, *, verify_init: bool = True
) -> ObjectBluePrint[T]:
    args, kwargs = infer_args(arg, verify_init=verify_init)

    spec: ObjectBluePrint[T] = {
        "__module_name__": arg.__class__.__module__,
        "__class_name__": arg.__class__.__qualname__,
        "__args__": args,
        "__kwargs__": kwargs,
    }

    assert is_object_blueprint(spec)
    return spec


def _infer_tensor_blueprint(tensor: Tensor) -> TensorBluePrint:
    return {
        "__tensor__": tensor,
        "__dtype__": str(tensor.dtype),
        "__shape__": list(tensor.shape),
    }


def _infer_model_blueprint[T: nn.Module](
    arg: T, /, *, verify_init: bool = True
) -> ModelBluePrint[T]:
    return _infer_object_blueprint(arg, verify_init=verify_init)


def infer_blueprint[T](arg: T, /) -> BluePrint[T]:
    return BLUEPRINT_REGISTRY.infer(arg)


def initialize[T](spec: BluePrint[T], /) -> T:
    if not is_blueprint(spec):
        raise TypeError("Expected a blueprint dictionary.")
    return BLUEPRINT_REGISTRY.initialize(spec)


def validate_blueprint[T](arg: T, spec: BluePrint[T], /) -> None:
    BLUEPRINT_REGISTRY.validate(arg, spec)


type BluePrintPredicate[T] = Callable[[Any], TypeGuard[BluePrint[T]]]
type BluePrintMaker[T] = Callable[[T], BluePrint[T]]
type BluePrintValidator[T] = Callable[[T, BluePrint[T]], None]
type BluePrintInitializer[T] = Callable[[BluePrint[T]], T]


class BluePrintRegistry:
    r"""Registry for blueprint initializers, validators and makers."""

    def __init__(self) -> None:
        self._initializers: list[tuple[BluePrintPredicate, BluePrintInitializer]] = []
        self.register(is_object_blueprint, _initialize_object_blueprint)
        self.register(is_model_blueprint, _initialize_object_blueprint)
        self.register(is_tensor_blueprint, _initialize_tensor_blueprint)

        self._validators: list[tuple[BluePrintPredicate, BluePrintValidator]] = []
        self.register_validator(is_object_blueprint, _validate_object_blueprint)
        self.register_validator(is_tensor_blueprint, _validate_tensor_blueprint)

        self._makers: dict[type, BluePrintMaker] = {}
        self.register_maker(nn.Module, _infer_model_blueprint)
        self.register_maker(Tensor, _infer_tensor_blueprint)

    def register[T](
        self,
        predicate: BluePrintPredicate[T],
        initializer: BluePrintInitializer[T],
        /,
    ) -> None:
        if any(existing is predicate for existing, _ in self._initializers):
            raise ValueError("Blueprint predicate is already registered.")
        self._initializers.append((predicate, initializer))

    def register_validator[T](
        self,
        predicate: BluePrintPredicate[T],
        validator: Callable[[T, BluePrint[T]], None],
        /,
    ) -> None:
        if any(existing is predicate for existing, _ in self._validators):
            raise ValueError("Blueprint predicate is already registered.")
        self._validators.append((predicate, validator))

    def register_maker[T](self, cls: type[T], maker: BluePrintMaker[T], /) -> None:
        if cls in self._makers:
            raise ValueError(f"Model class {cls.__qualname__} is already registered.")
        self._makers[cls] = maker

    def _select_initializer[T = Any](
        self, spec: BluePrint[T], /
    ) -> BluePrintInitializer[T]:
        for predicate, initializer in self._initializers:
            if predicate(spec):
                return initializer
        raise TypeError("Unsupported blueprint type.")

    def _select_validator[T = Any](
        self, spec: BluePrint[T], /
    ) -> BluePrintValidator[T]:
        for predicate, validator in self._validators:
            if predicate(spec):
                return validator
        raise TypeError("Unsupported blueprint type.")

    def _select_maker[T](self, arg: T, /) -> BluePrintMaker[T]:
        for cls, maker in self._makers.items():
            if isinstance(arg, cls):
                return maker
        return _infer_object_blueprint

    def initialize[T](self, spec: BluePrint[T], /, *, validate: bool = False) -> T:
        initializer = self._select_initializer(spec)
        result = initializer(spec)
        if validate:
            self.validate(result, spec)
        return result

    def validate[T](self, result: T, spec: BluePrint[T], /) -> None:
        validator = self._select_validator(spec)
        validator(result, spec)

    def infer[T](self, arg: T, /) -> BluePrint[T]:
        maker = self._select_maker(arg)
        return maker(arg)


BLUEPRINT_REGISTRY = BluePrintRegistry()


def _value_to_json(value: ArgValue, /) -> JSON_Value:
    match value:
        case Tensor():
            if value.numel() == 1:
                return _value_to_json(value.item())
            if value.ndim <= len(MAX_SHAPE) and value.shape <= MAX_SHAPE:
                return _value_to_json(value.tolist())
            raise NotImplementedError(
                f"Tensor shape {tuple(value.shape)!r} exceeds MAX_SHAPE."
            )
        case nn.Module():
            blueprint = infer_blueprint(value)
            return blueprint_to_json(blueprint)
        case list():
            return [_value_to_json(item) for item in value]
        case tuple():
            return [_value_to_json(item) for item in value]
        case dict():
            return {key: _value_to_json(item) for key, item in value.items()}
        case None | bool() | int() | float() | str():
            return value
        case _:
            raise TypeError(f"Unsupported argument value type: {type(value)!r}.")


def blueprint_to_json(arg: BluePrint, /) -> JSON:
    parsed = _value_to_json(cast("dict", arg))
    assert isinstance(parsed, dict)
    return parsed
