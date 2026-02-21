r"""Core types and functions for blueprint initialization and inference."""
# ruff: noqa: SIM103

__all__ = [
    # Constants
    "BLUEPRINT_REGISTRY",
    "INFER_ARGS_REGISTRY",
    # types
    "Makes",
    # registries
    "InferArgsRegistry",
    "BlueprintRegistry",
    # Blueprint types
    "Blueprint",
    "BasicBlueprint",
    "HydraBlueprint",
    "ObjectBlueprint",
    "is_blueprint",
    "is_blueprint_key",
    "is_basic_blueprint",
    "is_hydra_blueprint",
    "is_object_blueprint",
    # functions
    "blueprint_to_json",
    "infer_args",
    "infer_blueprint",
    "initialize",
    "validate_blueprint",
]

import inspect
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from inspect import Parameter
from typing import (
    Any,
    NotRequired,
    ReadOnly,
    TypeGuard,
    TypeIs,
    cast,
    overload,
)

from typing_extensions import TypedDict  # using extra_items (3.15)

from blueprint.config import SupportsFromConfig, has_config, is_config

type Makes[T] = T | type[T] | Blueprint[T]
type FilePath = str | os.PathLike[str]

# The types allowed in the config dictionary.
# configs should not contain ModelBlueprints or TensorSpecs directly, but actual models and tensors
type POArgs = list[object]
type KWArgs = dict[Identifier, object]
type Args = tuple[POArgs, KWArgs]

type JSON_Leaf = None | bool | int | float | str
type JSON_Value = JSON_Leaf | list[JSON_Value] | dict[str, JSON_Value]
type JSON = dict[str, JSON_Value]


# region key and value types -----------------------------------------------------------
type Key = str  # any string.
type Identifier = str  # identifier
type DunderKey = str  # identifier & dunder
type SunderKey = str  # identifier & sunder (not dunder)
type PrivateKey = str  # identifier & starts with _ (includes dunder and sunder)


def _is_dunder_key(value: object, /) -> TypeGuard[DunderKey]:
    r"""String that starts and ends with double underscores and is a valid identifier."""
    return (
        isinstance(value, str)
        and len(value) > 4
        and value.isidentifier()
        and value.startswith("__")
        and value.endswith("__")
        and not value.startswith("___")
        and not value.endswith("___")
    )


def _is_public_key(value: object, /) -> TypeGuard[Identifier]:
    r"""String that is a valid identifier and does not start with an underscore."""
    return isinstance(value, str) and value.isidentifier() and not value.startswith("_")


def _is_sunder_key(value: object, /) -> TypeGuard[SunderKey]:
    r"""Strings of the form `_key_` that are valid identifiers, but not dunder keys."""
    return (
        isinstance(value, str)
        and len(value) > 2
        and value.isidentifier()
        and value.startswith("_")
        and value.endswith("_")
        and not value.startswith("__")
        and not value.endswith("__")
    )


def is_blueprint_key(value: object, /) -> TypeGuard[DunderKey | SunderKey]:
    r"""Any Sunder/Dunder string is considered a blueprint key."""
    return _is_sunder_key(value) or _is_dunder_key(value)


# endregion allowed types --------------------------------------------------------------


# region blueprint types ---------------------------------------------------------------
class Blueprint[T](TypedDict, extra_items=ReadOnly[object]):  # type: ignore[call-arg]
    r"""A dictionary that can be used to initialize an object of type T.

    All keys must be dunder or sunder.
    """

    # dunder and sunder keys only (not expressible in the type system)


class BasicBlueprint[T](Blueprint[T]):
    r"""A basic blueprint that only contains the module and class name."""

    __name__: ReadOnly[str]
    __module__: ReadOnly[str]
    # **InitKey: kwargs to pass to the model
    # **DunderKey: object (reserved for future use)
    # **SunderKey: object (reserved for future use)


class ObjectBlueprint[T](Blueprint[T]):
    r"""A dictionary that allows initializing an object."""

    __module_name__: ReadOnly[str]
    __class_name__: ReadOnly[str]
    __module_version__: NotRequired[ReadOnly[str]]

    __args__: ReadOnly[list[object]]
    __kwargs__: ReadOnly[dict[Identifier, object]]
    # **DunderKey: object (reserved for future use)


class HydraBlueprint[T](TypedDict):
    r"""A blueprint compatible with Hydra's instantiation syntax."""

    _target_: ReadOnly[str]
    _args_: NotRequired[ReadOnly[list[object]]]


def is_blueprint(arg: object, /) -> TypeGuard[Blueprint]:
    if not isinstance(arg, dict):
        return False
    if not any(is_blueprint_key(key) for key in arg):
        return False
    return True


def is_object_blueprint(arg: object, /) -> TypeGuard[ObjectBlueprint]:
    if not is_blueprint(arg):
        return False
    if not ObjectBlueprint.__required_keys__.issubset(arg.keys()):
        return False
    if not isinstance(arg.get("__module_name__"), str):
        return False
    if not isinstance(arg.get("__class_name__"), str):
        return False
    if not (isinstance(arg.get("__args__"), list)):
        return False
    if not (
        isinstance(kwargs := arg.get("__kwargs__"), dict)
        and all(_is_public_key(key) for key in kwargs)
    ):
        return False

    return True


def is_hydra_blueprint(arg: object, /) -> TypeGuard[HydraBlueprint]:
    if not is_blueprint(arg):
        return False
    if not HydraBlueprint.__required_keys__.issubset(arg.keys()):
        return False
    if not isinstance(arg.get("_target_"), str):
        return False
    if not (isinstance(arg.get("_args_"), None | list)):
        return False
    # all keys should be sunder_keys or public_keys.
    if not all(_is_sunder_key(key) or _is_public_key(key) for key in arg):
        return False
    return True


def is_basic_blueprint(obj: object, /) -> TypeIs[BasicBlueprint]:
    r"""Check if the object is a configuration dictionary.

    A configuration is mapping with the following keys:

    - `__module__` (`str`): The module name.
    - `__name__` (`str`): The class name.
    - `__args__` (`Sequence`, optional): The positional arguments for the class.
    - any extra keys that are valid, non-private identifiers.

    """
    if not isinstance(obj, Mapping):
        return False
    if not isinstance(obj.get("__module__"), str):
        return False
    if not isinstance(obj.get("__name__"), str):
        return False
    if not isinstance(obj.get("__args__"), None | Sequence):
        return False

    # check that extra keys are valid identifiers
    for key in obj:
        if not isinstance(key, str):
            return False
        if key in {"__module__", "__name__", "__args__"}:
            continue
        if not key.isidentifier() or key.startswith("_"):
            return False

    # check JSON serializability
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
    if any(_is_dunder_key(name) for name in kwargs):
        raise ValueError("Parameter names cannot be dunder names.")

    if missing := {
        name
        for name in kwargs
        if not hasattr(arg, name) and kwargs[name].default is Parameter.empty
    }:
        raise ValueError(f"Missing model attributes for config: {sorted(missing)}")

    return [], {name: getattr(arg, name) for name in (kwargs.keys() - missing)}


class InferArgsRegistry(Mapping[type, Callable[[Any], Args]]):
    r"""Registry for functions that infer model args and kwargs from instances."""

    def __init__(self) -> None:
        self._registry: dict[type, Callable[[Any], Args]] = {}

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
            m = _initialize_from_args(cls, args, kwargs)
        except Exception as exc:
            raise ValueError(
                f"Failed to verify inferred config for {cls.__qualname__}."
            ) from exc
        assert type(m) is cls

    assert is_config(kwargs)
    return args, kwargs


def _initialize_from_args[T](cls: type[T], args: POArgs, kwargs: KWArgs, /) -> T:
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


def _resolve_value(arg: object, /) -> Any:
    match arg:
        case blueprint if is_blueprint(blueprint):
            return initialize(blueprint)
        case list():
            return [_resolve_value(item) for item in arg]
        case tuple():
            return tuple(_resolve_value(item) for item in arg)
        case dict():
            return {key: _resolve_value(item) for key, item in arg.items()}
        case _:
            return arg


def _validate_object_blueprint[T](arg: T | type[T], spec: Blueprint, /) -> None:
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


def _infer_object_blueprint[T](
    arg: T, /, *, verify_init: bool = True
) -> ObjectBlueprint[T]:
    args, kwargs = infer_args(arg, verify_init=verify_init)

    spec: ObjectBlueprint[T] = {
        "__module_name__": arg.__class__.__module__,
        "__class_name__": arg.__class__.__qualname__,
        "__args__": args,
        "__kwargs__": kwargs,
    }

    assert is_object_blueprint(spec)
    return spec


def _initialize_object[T](spec: Blueprint[T], /) -> T:
    if not is_object_blueprint(spec):
        raise TypeError("Expected an object blueprint dictionary.")

    module_name = spec["__module_name__"]
    class_name = spec["__class_name__"]

    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    args = [_resolve_value(item) for item in spec["__args__"]]
    kwargs = {key: _resolve_value(item) for key, item in spec["__kwargs__"].items()}

    return _initialize_from_args(cls, args, kwargs)


@overload
def _is_json(value: Blueprint, /) -> TypeGuard[JSON]: ...
@overload
def _is_json(value: object, /) -> TypeGuard[JSON_Value]: ...
def _is_json(value: object, /) -> TypeGuard[JSON_Value]:
    match value:
        case list():
            return all(_is_json(item) for item in value)
        case dict(mapping):
            return all(
                isinstance(key, str) and _is_json(item) for key, item in mapping.items()
            )
        case None | bool() | int() | float() | str():
            return True
        case _:
            return False


@overload
def _naive_serializer(value: Blueprint, /) -> JSON: ...
@overload
def _naive_serializer(value: object, /) -> JSON_Value: ...
def _naive_serializer(value: object, /) -> JSON_Value:
    match value:
        case None:
            return None
        case bool():
            return bool(value)
        case int():
            return int(value)
        case float():
            return float(value)
        case str():
            return str(value)
        case list():
            return [_naive_serializer(item) for item in value]
        case tuple():
            return [_naive_serializer(item) for item in value]
        case dict():
            return {str(key): _naive_serializer(item) for key, item in value.items()}
        case other:
            try:
                blueprint = infer_blueprint(other)
            except Exception as exc:
                exc.add_note(
                    f"Failed to infer blueprint for value of type {type(other)}."
                )
                raise
            return blueprint_to_json(blueprint)


type BlueprintPredicate[T] = Callable[[Any], TypeGuard[Blueprint[T]]]
type BlueprintMaker[T] = Callable[[T], Blueprint[T]]
type BlueprintValidator[T] = Callable[[T, Blueprint[T]], None]
type BlueprintInitializer[T] = Callable[[Blueprint[T]], T]
type BlueprintSerializer[T] = Callable[[Blueprint[T]], JSON]


class BlueprintRegistry:
    r"""Registry for blueprint initializers, validators and makers."""

    def __init__(self) -> None:
        self._blueprints: dict[type[Blueprint], object] = {}

        self._initializers: list[tuple[BlueprintPredicate, BlueprintInitializer]] = []
        self._validators: list[tuple[BlueprintPredicate, BlueprintValidator]] = []
        self._serializers: dict[BlueprintPredicate, BlueprintSerializer] = {}

        self._makers: dict[type, BlueprintMaker] = {}

    def register_blueprint[T](self, cls: type[Blueprint[T]], /) -> None:
        if cls in self._blueprints:
            raise ValueError(
                f"Blueprint class {cls.__qualname__} is already registered."
            )
        self._blueprints[cls] = None

    def register_initializer[T](
        self,
        predicate: BlueprintPredicate[T],
        initializer: BlueprintInitializer[T],
        /,
    ) -> None:
        if any(existing is predicate for existing, _ in self._initializers):
            raise ValueError("Blueprint predicate is already registered.")
        self._initializers.append((predicate, initializer))

    def register_validator[T](
        self,
        predicate: BlueprintPredicate[T],
        validator: Callable[[T, Blueprint[T]], None],
        /,
    ) -> None:
        if any(existing is predicate for existing, _ in self._validators):
            raise ValueError("Blueprint predicate is already registered.")
        self._validators.append((predicate, validator))

    def register_maker[T](self, cls: type[T], maker: BlueprintMaker[T], /) -> None:
        if cls in self._makers:
            raise ValueError(f"Model class {cls.__qualname__} is already registered.")
        self._makers[cls] = maker

    def register_serializer[T](
        self, predicate: BlueprintPredicate[T], serializer: BlueprintSerializer[T], /
    ) -> None:
        if any(existing is predicate for existing, _ in self._serializers.items()):
            raise ValueError("Blueprint predicate is already registered.")
        self._serializers[predicate] = serializer

    def _select_initializer[T = Any](
        self, spec: Blueprint[T], /
    ) -> BlueprintInitializer[T]:
        for predicate, initializer in self._initializers:
            if predicate(spec):
                return initializer
        raise TypeError("Unsupported blueprint type.")

    def _select_validator[T = Any](
        self, spec: Blueprint[T], /
    ) -> BlueprintValidator[T]:
        for predicate, validator in self._validators:
            if predicate(spec):
                return validator
        raise TypeError("Unsupported blueprint type.")

    def _select_serializer[T](self, spec: Blueprint[T], /) -> BlueprintSerializer[T]:
        for predicate, serializer in self._serializers.items():
            if predicate(spec):
                return serializer
        return _naive_serializer

    def _select_maker[T](self, arg: T, /) -> BlueprintMaker[T]:
        for cls, maker in self._makers.items():
            if isinstance(arg, cls):
                return maker
        return _infer_object_blueprint

    def initialize[T](self, spec: Blueprint[T], /, *, validate: bool = False) -> T:
        initializer = self._select_initializer(spec)
        result = initializer(spec)
        if validate:
            self.validate(result, spec)
        return result

    def validate[T](self, result: T, spec: Blueprint[T], /) -> None:
        validator = self._select_validator(spec)
        validator(result, spec)

    def infer[T](self, arg: T, /) -> Blueprint[T]:
        maker = self._select_maker(arg)
        return maker(arg)

    def serialize[T](self, spec: Blueprint[T], /) -> JSON:
        if _is_json(spec):
            return spec

        serializer = self._select_serializer(spec)
        return serializer(spec)


BLUEPRINT_REGISTRY = BlueprintRegistry()
BLUEPRINT_REGISTRY.register_initializer(is_object_blueprint, _initialize_object)
BLUEPRINT_REGISTRY.register_validator(is_object_blueprint, _validate_object_blueprint)
# BLUEPRINT_REGISTRY.register_serializer(is_blueprint, _object_to_json)


def blueprint_to_json[T](arg: Blueprint[T], /) -> JSON:
    return BLUEPRINT_REGISTRY.serialize(arg)


def infer_blueprint[T](arg: T, /) -> Blueprint[T]:
    return BLUEPRINT_REGISTRY.infer(arg)


def validate_blueprint[T](arg: T, spec: Blueprint[T], /) -> None:
    BLUEPRINT_REGISTRY.validate(arg, spec)


def initialize[T = Any](spec: T | type[T] | Blueprint[T], /) -> T:
    r"""Initialize an object from a blueprint or return the object if it's not a blueprint.

    Args:
        spec: The blueprint to initialize from, or the object to return if it's not a blueprint.
    """
    if isinstance(spec, type):
        return _initialize_from_args(cast("type[T]", spec), [], {})
    if isinstance(spec, dict):
        if not is_blueprint(spec):
            raise TypeError("Expected a blueprint dictionary.")
        return BLUEPRINT_REGISTRY.initialize(spec)
    # fall through
    return spec
