r"""Demonstration of model configuration inference and export."""

import inspect
import json
import os
from collections.abc import Callable, Iterator, Mapping
from importlib import metadata
from inspect import Parameter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import (
    Any,
    ClassVar,
    Protocol,
    ReadOnly,
    Self,
    TypedDict,
    TypeGuard,
    TypeIs,
    runtime_checkable,
)
from zipfile import ZipFile

import torch
from torch import Tensor, nn
from torch.export import ExportedProgram

SPEC_VERSION = "1.0"


type Makes[T] = type[T] | ModelSpec | dict[str, Any] | object  # dataclass
type Activation = Callable[[Tensor], Tensor]

type FilePath = str | os.PathLike[str]
type Hyperparameters = dict[str, Any]


class TensorSpec(TypedDict):
    __tensor__: str
    path: str
    dtype: str
    shape: list[int]


ALLOWED_TYPES = (
    str,
    int,
    float,
    bool,
    type(None),
    Tensor,
    tuple,
    list,
    dict,
    nn.Module,
)

type Key = str  # identifier
type InitKey = str  # identifier & not dunder
type DunderKey = str  # identifier & dunder


# The types allowed in the config dictionary.
# configs should not contain ModelSpecs or TensorSpecs directly, but actual models and tensors
type ArgLeaf = None | bool | int | float | str | Tensor | nn.Module
type ArgValue = ArgLeaf | list[ArgValue] | tuple[ArgValue, ...] | dict[Key, ArgValue]
type POArgs = list[ArgValue]
type KWArgs = dict[InitKey, ArgValue]
type Args = tuple[POArgs, KWArgs]
type Config = KWArgs  # Alias
r"""Only contains non-dunder keys and values of allowed types."""

# FIXME: do we need an intermediate Spec type for in memory spec, where only
# models are replaces with ModelSpec?

# The Types allowed in serialized specs. (no actual models or tensors, only specs)
type JSON_Leaf = None | bool | int | float | str | TensorSpec | ModelSpec
type JSON_Value = JSON_Leaf | list[JSON_Value] | dict[Key, JSON_Value]
type JSON = dict[Key, JSON_Value]

type SpecArgs = list[JSON_Value]
type SpecKWArgs = dict[Key, JSON_Value]


class ModelSpec(TypedDict):
    r"""A dictionary that allows initializing a model."""

    __module_name__: ReadOnly[str]
    __class_name__: ReadOnly[str]

    __args__: ReadOnly[SpecArgs]
    __kwargs__: ReadOnly[SpecKWArgs]
    # **DunderKey: object (reserved for future use)


class SerializedModelSpec(ModelSpec, TypedDict):
    # __module_name__: ReadOnly[str]
    # __class_name__: ReadOnly[str]
    #
    # __args__: ReadOnly[SpecArgs]
    # __kwargs__: ReadOnly[SpecKWArgs]

    # new keys for serialized models
    __module_version__: ReadOnly[str]
    __spec_version__: ReadOnly[str]

    __storage_path__: ReadOnly[str]
    __storage_format__: ReadOnly[str]
    # **DunderKey: object (reserved for future use)


def is_dunder(value: str, /) -> TypeGuard[DunderKey]:
    return (
        len(value) > 4
        and value.startswith("__")
        and value.endswith("__")
        and not value.startswith("___")
        and not value.endswith("___")
    )


def is_scalar(arg: object, /) -> TypeIs[ArgLeaf]:
    return arg is None or isinstance(arg, (bool, int, float, str))


def is_config_key(arg: object, /) -> TypeGuard[Key]:
    return isinstance(arg, str) and arg.isidentifier() and not is_dunder(arg)


def is_config_value(arg: object, /) -> TypeIs[ArgValue]:
    match arg:
        case None | bool() | int() | float() | str() | Tensor() | nn.Module():
            return True
        case list() | tuple():
            return all(is_config_value(value) for value in arg)
        case dict():
            return all(
                is_config_key(key) and is_config_value(value)
                for key, value in arg.items()
            )
        case _:
            return False


def is_config(arg: object, /) -> TypeIs[Config]:
    if not isinstance(arg, dict):
        return False
    return all(
        is_config_key(key) and is_config_value(value) for key, value in arg.items()
    )


class ModelRegistry(Mapping[type[nn.Module], Callable[[nn.Module], Args]]):
    def __init__(self) -> None:
        self._registry: dict[type[nn.Module], Callable[[nn.Module], Args]] = {}
        self.register(nn.Sequential, self.infer_nn_Sequential)
        self.register(nn.ModuleList, self.infer_nn_ModuleList)
        self.register(nn.ModuleDict, self.infer_nn_ModuleDict)
        self.register(nn.Linear, self.infer_nn_Linear)

    def __getitem__[T: nn.Module](self, key: type[T], /) -> Callable[[T], Args]:
        return self._registry[key]

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self) -> Iterator[type[nn.Module]]:
        return iter(self._registry)

    def register[T: nn.Module](
        self,
        model_cls: type[T],
        infer_fn: Callable[[T], Args],
    ) -> None:
        if model_cls in self._registry:
            raise ValueError(
                f"Model class {model_cls.__qualname__} is already registered."
            )
        self._registry[model_cls] = infer_fn

    def infer_nn_Linear(self, model: nn.Linear) -> Args:
        return [], {
            "in_features": model.in_features,
            "out_features": model.out_features,
            "bias": model.bias is not None,
        }

    def infer_nn_Sequential(self, model: nn.Sequential) -> Args:
        return list(model), {}

    def infer_nn_ModuleList(self, model: nn.ModuleList) -> Args:
        return [list(model)], {}

    def infer_nn_ModuleDict(self, model: nn.ModuleDict) -> Args:
        return [dict(model)], {}


REGISTRY = ModelRegistry()


def _infer_config_from_init(model: nn.Module, /) -> Args:
    if (infer_fn := REGISTRY.get(type(model))) is not None:
        return infer_fn(model)

    signature = inspect.signature(model.__init__)
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
    if any(is_dunder(name) for name in kwargs):
        raise ValueError("Parameter names cannot be dunder names.")

    if missing := {
        name
        for name in kwargs
        if not hasattr(model, name) and kwargs[name].default is Parameter.empty
    }:
        raise ValueError(f"Missing model attributes for config: {sorted(missing)}")

    return [], {name: getattr(model, name) for name in (kwargs.keys() - missing)}


def infer_config(model: nn.Module, *, verify_init: bool = True) -> Args:
    r"""Infer the configuration dictionary for a model.

    Args:
        model: The model to infer the configuration from.
        verify_init: Whether to verify that the inferred config can be used
            to initialize the model.
    """
    args: POArgs
    kwargs: KWArgs
    if has_config(model):
        args, kwargs = [], model.config
    else:
        args, kwargs = _infer_config_from_init(model)

    if verify_init:
        model_cls = type(model)
        try:
            if issubclass(model_cls, SupportsFromConfig):
                assert not args
                model_cls.from_config(kwargs)
            else:
                model_cls(*args, **kwargs)
        except Exception as exc:
            raise ValueError(
                f"Failed to initialize {model_cls.__qualname__} from inferred config."
            ) from exc

    assert is_config(kwargs)
    return args, kwargs


def is_spec_key(arg: object, /) -> TypeGuard[DunderKey]:
    return isinstance(arg, str) and arg.isidentifier() and is_dunder(arg)


def is_spec_value(arg: object, /) -> TypeIs[JSON_Value]:
    match arg:
        case None | bool() | int() | float() | str():
            return True
        case list() | tuple():
            return all(is_spec_value(value) for value in arg)
        case dict():
            if any(is_dunder(key) for key in arg):
                return is_tensor_spec(arg) or is_model_spec(arg)
            return all(
                is_config_key(key) and is_spec_value(value)
                for key, value in arg.items()
            )
        case _:
            return False


def is_model_spec(arg: object, /) -> TypeIs[ModelSpec]:
    if not isinstance(arg, dict):
        return False
    if not ModelSpec.__required_keys__.issubset(arg.keys()):
        return False
    if not isinstance(arg.get("__module_name__"), str):
        return False
    if not isinstance(arg.get("__class_name__"), str):
        return False
    if not isinstance(arg.get("__args__"), list):
        return False
    if not isinstance(arg.get("__kwargs__"), dict):
        return False
    if not all(is_spec_value(item) for item in arg["__args__"]):
        return False
    if not all(
        is_config_key(key) and is_spec_value(value)
        for key, value in arg["__kwargs__"].items()
    ):
        return False
    if not all(is_spec_key(key) for key in arg):
        return False
    return True


def is_serialized_model_spec(arg: object, /) -> TypeIs[SerializedModelSpec]:
    if not is_model_spec(arg):
        return False
    if not SerializedModelSpec.__required_keys__.issubset(arg.keys()):
        return False
    if not isinstance(arg.get("__storage_format__"), str):
        return False
    if not isinstance(arg.get("__storage_path__"), str):
        return False
    return True


def config_to_spec(value: ArgValue, *, verify_init: bool = True) -> JSON_Value:
    match value:
        case nn.Module():
            return infer_modelspec(value, verify_init=verify_init)
        case list():
            return [config_to_spec(item) for item in value]
        case tuple():
            return tuple(config_to_spec(item) for item in value)
        case dict():
            return {key: config_to_spec(item) for key, item in value.items()}
        case _:
            return value


def infer_modelspec(model: nn.Module, *, verify_init: bool = True) -> ModelSpec:
    args, kwargs = infer_config(model, verify_init=verify_init)
    converted_args = config_to_spec(args, verify_init=verify_init)
    converted_kwargs = config_to_spec(kwargs, verify_init=verify_init)

    spec: ModelSpec = {
        "__module_name__": model.__class__.__module__,
        "__class_name__": model.__class__.__qualname__,
        "__args__": converted_args,
        "__kwargs__": converted_kwargs,
    }

    assert is_model_spec(spec)
    return spec


class SupportsConfig(Protocol):
    r"""Models that support a hyperparameter dictionary.

    A hyperparameter dictionary should be such that

    type(model)(**model.config)

    recovers the model.
    """

    @property
    def config(self) -> Config: ...


class SupportsDefaultConfig(Protocol):
    r"""Models that have a default configuration dataclass."""

    DEFAULT_CONFIG: ClassVar[type]


def has_config(arg: object, /) -> TypeIs[SupportsConfig]:
    if (config := getattr(arg, "config", None)) is None:
        return False
    return is_config(config)


def has_default_config(arg: object, /) -> TypeIs[SupportsDefaultConfig]:
    if not isinstance(arg, type):
        arg = type(arg)
    default_config = getattr(arg, "DEFAULT_CONFIG", None)
    return default_config is not None


@runtime_checkable
class SupportsFromConfig(Protocol):
    r"""Models that can be explicitly initialized from a configuration dictionary."""

    @classmethod
    def from_config(cls, config: Config, /) -> Self: ...


def _import_value(arg: JSON_Value, /) -> object:
    match arg:
        case _ if is_tensor_spec(arg):
            return import_tensor(arg)
        case _ if is_model_spec(arg):
            return import_model(arg)
        case list():
            return [_import_value(item) for item in arg]
        case tuple():
            return tuple(_import_value(item) for item in arg)
        case dict():
            return {key: _import_value(item) for key, item in arg.items()}
        case _:
            return arg


def initialize_model_from_spec(spec: ModelSpec) -> nn.Module:
    if not is_model_spec(spec):
        raise TypeError("Expected a model spec dictionary.")

    module_name = spec["__module_name__"]
    class_name = spec["__class_name__"]

    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    args = [_import_value(item) for item in spec["__args__"]]
    kwargs = {key: _import_value(item) for key, item in spec["__kwargs__"].items()}

    if issubclass(cls, SupportsFromConfig):
        if args:
            raise ValueError("from_config does not support positional arguments.")
        return cls.from_config(kwargs)
    return cls(*args, **kwargs)


def import_model(spec: ModelSpec, /) -> nn.Module:
    if is_serialized_model_spec(spec):
        return import_model_from_spec(spec)

    model = initialize_model_from_spec(spec)
    validate_model_spec(model, spec)

    return model


def serialize_model(
    model: nn.Module | ExportedProgram, filepath: FilePath
) -> SerializedModelSpec:
    spec = infer_modelspec(model)
    path = Path(filepath)
    # ensure path ends with .pt or .zip
    if path.suffix not in (".pt", ".zip"):
        raise ValueError("Model file extension must be .pt or .zip")

    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        with archive.open("model.pt", "w") as model_file:
            match model:
                case torch.jit.RecursiveScriptModule():
                    fmt = "torchscript"
                    torch.jit.save(model, model_file)
                case ExportedProgram():
                    fmt = "torch_export"
                    torch.export.save(model, model_file)
                case nn.Module():
                    fmt = "state_dict"
                    torch.save(model.state_dict(), model_file)
                case _:
                    raise TypeError(f"Expected nn.Module, got {type(model)}")

        # TODO: replace any tensors in the spec with tensor specs,
        #  and save them in assets/initialization/*.pt

        spec = spec | {
            "__storage_path__": str(path),
            "__storage_format__": fmt,
            "__spec_version__": SPEC_VERSION,
            "__module_version__": _infer_module_version(model.__class__.__module__),
        }
        archive.writestr("config.json", json.dumps(spec))

    return spec


def deserialize_model(path: FilePath) -> nn.Module:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    if path.suffix not in (".pt", ".zip"):
        raise ValueError("Model file extension must be .pt or .zip")

    with (
        ZipFile(path, "r") as archive,
        archive.open("config.json", "r") as config_file,
    ):
        spec = json.load(config_file)

    assert is_serialized_model_spec(spec)
    return import_model_from_spec(spec)


def import_model_from_spec(spec: SerializedModelSpec) -> nn.Module:
    if not is_serialized_model_spec(spec):
        raise TypeError("Expected a serialized model spec dictionary.")

    fmt = spec["__storage_format__"]
    path = Path(spec["__storage_path__"])

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    if path.suffix not in (".pt", ".zip"):
        raise ValueError("Model file extension must be .pt or .zip")

    with (
        ZipFile(path, "r") as archive,
        archive.open("model.pt", "r") as model_file,
    ):
        match fmt:
            case "torchscript":
                module = torch.jit.load(model_file)
                assert isinstance(module, nn.Module)
                return module
            case "torch_export":
                imported = torch.export.load(model_file)
                return imported.module()
            case "state_dict":
                instance = initialize_model_from_spec(spec)
                state = torch.load(model_file)
                instance.load_state_dict(state)
                return instance
            case _:
                raise ValueError(f"Unsupported model format: {fmt!r}")


def validate_model_spec(model: nn.Module | type[nn.Module], spec: ModelSpec) -> None:
    if not is_model_spec(spec):
        raise TypeError("Invalid model spec.")
    if any(not key.isidentifier() for key in spec):
        raise TypeError("Model spec keys must be identifiers.")

    model_cls = model if isinstance(model, type) else model.__class__
    assert issubclass(model_cls, nn.Module)

    expected_module = spec["__module_name__"]
    expected_name = spec["__class_name__"]

    if model_cls.__module__ != expected_module:
        raise ValueError(
            f"Module path mismatch: expected {expected_module},"
            f" got {model_cls.__module__}"
        )
    if model_cls.__qualname__ != expected_name:
        raise ValueError(
            f"Module name mismatch: expected {expected_name},"
            f" got {model_cls.__qualname__}"
        )

    if isinstance(model, nn.Module) and has_config(model):
        hp_keys = set(model.config.keys())
        spec_keys = set(spec["__kwargs__"].keys())
        missing = spec_keys - hp_keys
        if missing:
            raise ValueError(f"Model spec keys not present in HP: {sorted(missing)}")


def _infer_module_version(arg: str | ModuleType) -> str | None:
    module_name = None
    match arg:
        case str():
            module_name = arg
        case ModuleType():
            module_name = arg.__name__
        case _:
            raise TypeError(f"Unsupported module type: {type(arg)}")

    root_name = module_name.split(".", 1)[0]

    try:
        return metadata.version(root_name)
    except Exception:
        pass

    if isinstance(arg, ModuleType):
        module = arg
    else:
        try:
            module = __import__(arg)
        except Exception:
            return None

    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def is_tensor_spec(value: object, /) -> TypeIs[TensorSpec]:
    if not isinstance(value, dict):
        return False
    required = set(TensorSpec.__required_keys__)
    if not required.issubset(value.keys()):
        return False
    return all(isinstance(key, str) and key.isidentifier() for key in value)


def validate_tensor_spec(tensor: Tensor, spec: TensorSpec) -> None:
    expected_dtype = spec["dtype"]
    expected_shape = spec["shape"]
    if str(tensor.dtype) != expected_dtype:
        raise ValueError(
            f"Tensor dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
        )
    if list(tensor.shape) != list(expected_shape):
        raise ValueError(
            f"Tensor shape mismatch: expected {expected_shape}, got {list(tensor.shape)}"
        )


def spec_from_tensor(tensor: Tensor, path: FilePath) -> TensorSpec:
    return {
        "__tensor__": "torch",
        "path": str(path),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }


def import_tensor(spec: TensorSpec) -> Tensor:
    tensor = torch.load(spec["path"])
    validate_tensor_spec(tensor, spec)
    return tensor


def export_tensor(tensor: Tensor, path: FilePath) -> TensorSpec:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    return {
        "__tensor__": "torch",
        "path": str(path),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }


class TestInitialization:
    def test_linear_model(self) -> None:
        model = nn.Linear(4, 8, bias=False)
        spec = infer_modelspec(model)
        validate_model_spec(model, spec)
        clone = initialize_model_from_spec(spec)
        assert isinstance(clone, nn.Linear)
        assert clone.in_features == 4
        assert clone.out_features == 8
        assert clone.bias is None

    def test_sequence_model(self) -> None:
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
        )
        spec = infer_modelspec(model)
        validate_model_spec(model, spec)
        clone = initialize_model_from_spec(spec)
        assert isinstance(clone, nn.Sequential)
        assert len(clone) == 2
        assert isinstance(clone[0], nn.Linear)
        assert clone[0].in_features == 4
        assert clone[0].out_features == 8
        assert isinstance(clone[1], nn.ReLU)


class TestSerialization:
    def test_linear(self) -> None:
        model = nn.Linear(4, 8, bias=False)
        spec = infer_modelspec(model)
        validate_model_spec(model, spec)
        serialize_model(model, "model.zip")
        deserialized = deserialize_model("model.zip")
        assert isinstance(deserialized, nn.Linear)
        assert deserialized.in_features == 4
        assert deserialized.out_features == 8
        assert deserialized.bias is None

    def test_sequential(self) -> None:
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
        )
        spec = infer_modelspec(model)
        validate_model_spec(model, spec)

        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            spec_path = Path(tmpdir) / "config.json"
            serialized_spec = serialize_model(model, model_path)

            with spec_path.open("w", encoding="utf-8") as file:
                json.dump(serialized_spec, file)
            with spec_path.open("r", encoding="utf-8") as file:
                deserialized_spec = json.load(file)

            deserialized = deserialize_model(model_path)
            # deserialized = import_model_from_spec(deserialized_spec)

        assert isinstance(deserialized, nn.Sequential)
        assert len(deserialized) == 2
        assert isinstance(deserialized[0], nn.Linear)
        assert deserialized[0].in_features == 4
        assert deserialized[0].out_features == 8
        assert isinstance(deserialized[1], nn.ReLU)
        assert isinstance(original_weight := model[0].weight, Tensor)
        assert isinstance(deserialized_weight := deserialized[0].weight, Tensor)
        assert torch.equal(original_weight, deserialized_weight)
