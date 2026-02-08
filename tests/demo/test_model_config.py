r"""Demonstration of model configuration inference and export."""

import inspect
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import (
    Any,
    NotRequired,
    Protocol,
    ReadOnly,
    Self,
    TypedDict,
    TypeIs,
    runtime_checkable,
)

import torch
from torch import Tensor, nn
from torch.export import ExportedProgram

SPEC_VERSION = "1.0"


type Makes[T] = type[T] | ModelSpec | dict[str, Any] | object  # dataclass
type Activation = Callable[[Tensor], Tensor]

type FilePath = str | os.PathLike[str]
type Hyperparameters = dict[str, Any]


class SerializedTensorSpec(TypedDict):
    __tensor__: str
    path: str
    dtype: str
    shape: list[int]


type Leaf = None | bool | int | float | str | Tensor | nn.Module
type C = Leaf | list[C] | tuple[C, ...] | dict[str, C]
type Config = dict[str, C]
r"""Only contains non-dunder keys and values of allowed types."""


class ModelSpec(TypedDict):
    __module__: ReadOnly[str]
    __name__: ReadOnly[str]
    __spec_version__: NotRequired[ReadOnly[str]]
    __module_version__: NotRequired[ReadOnly[str | None]]
    # **config parameters**


class SerializedModelSpec(TypedDict):
    __module__: ReadOnly[str]
    __name__: ReadOnly[str]
    __format__: ReadOnly[str]
    __spec_version__: ReadOnly[str]
    __module_version__: ReadOnly[str | None]
    __state_dict__: ReadOnly[str | None]
    # **config parameters**


class SupportsConfig(Protocol):
    r"""Models that support a hyperparameter dictionary.

    A hyperparameter dictionary should be such that

    type(model)(**model.config)

    recovers the model.
    """

    @property
    def config(self) -> Config: ...


@runtime_checkable
class SupportsFromConfig(Protocol):
    r"""Models that can be explicitly initialized from a configuration dictionary."""

    @classmethod
    def from_config(cls, config: Config, /) -> Self: ...


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


def _validate_plain_config(config: Config) -> None:
    for key in config:
        if not isinstance(key, str):
            raise TypeError(f"Config keys must be str, got {type(key).__name__}")
        if not key.isidentifier():
            raise ValueError(f"Config keys must be identifiers, got {key!r}")
    for value in config.values():
        match value:
            case None | bool() | int() | float() | str() | Tensor() | nn.Module():
                pass
            case list() | tuple():
                pass
            case dict(subconfig):
                _validate_plain_config(subconfig)
            case _:
                raise TypeError


def _infer_config_from_init(model: nn.Module, /) -> Config:
    init = model.__init__
    signature = inspect.signature(init)
    params = list(signature.parameters.values())

    if len(params) == 0:
        raise ValueError("illegal signature (no self parameter)")

    if any(p.kind is inspect.Parameter.POSITIONAL_ONLY for p in params):
        raise ValueError("Positional-only parameters are not supported")

    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        raise ValueError("*args parameters are not supported")

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        raise ValueError("**kwargs parameters are not supported")

    # skip 'self'
    param_names = [p.name for p in params[1:]]

    # validate names
    if not all(name.isidentifier() for name in param_names):
        raise ValueError("All parameter names must be valid identifiers.")
    if any(is_dunder(name) for name in param_names):
        raise ValueError("Parameter names cannot be dunder names.")

    if missing := [n for n in param_names if not hasattr(model, n)]:
        raise ValueError(f"Missing model attributes for config: {sorted(missing)}")

    return {name: getattr(model, name) for name in param_names}


def infer_config(
    model: nn.Module,
    *,
    verify_init: bool = True,
) -> dict[str, Any]:
    r"""Infer the configuration dictionary for a model.

    Args:
        model: The model to infer the configuration from.
        recursive: Whether to recursively infer configs for submodules.
        verify_init: Whether to verify that the inferred config can be used
            to initialize the model.
    """
    if (cfg := getattr(model, "config", None)) is None:
        config = _infer_config_from_init(model)
    else:
        config = cfg

    if not verify_init:
        return config

    model_cls = type(model)
    try:
        if issubclass(model_cls, SupportsFromConfig):
            model_cls.from_config(config)
        else:
            model_cls(**config)
    except Exception as exc:
        raise ValueError(
            f"Failed to initialize {model_cls.__qualname__} from inferred config."
        ) from exc
    return config


def infer_spec(model: nn.Module, *, verify_init: bool = True) -> ModelSpec:
    config = infer_config(model, verify_init=verify_init)

    spec: ModelSpec = {
        "__module__": model.__class__.__module__,
        "__name__": model.__class__.__qualname__,
        "__spec_version__": SPEC_VERSION,
        "__module_version__": _infer_module_version(model.__class__.__module__),
    } | config

    # recursively substitute submodules with their specs
    for name, value in spec.items():
        if isinstance(value, nn.Module):
            config[name] = infer_spec(value, verify_init=verify_init)

    return spec


def supports_config(module: object, /) -> TypeIs[SupportsConfig]:
    if not hasattr(module, "DEFAULT_CONFIG") or not hasattr(module, "HP"):
        return False
    hp = module.HP
    return is_model_spec(hp)


def is_dunder(value: str, /) -> bool:
    return (
        len(value) > 4
        and value.startswith("__")
        and value.endswith("__")
        and not value.startswith("___")
        and not value.endswith("___")
    )


def is_tensor_spec(value: object, /) -> TypeIs[SerializedTensorSpec]:
    if not isinstance(value, dict):
        return False
    required = set(SerializedTensorSpec.__required_keys__)
    if not required.issubset(value.keys()):
        return False
    return all(isinstance(key, str) and key.isidentifier() for key in value)


def is_scalar(value: object, /) -> bool:
    return value is None or isinstance(value, (bool, int, float, str, complex))


def is_model_spec(value: object, /) -> TypeIs[ModelSpec]:
    if not isinstance(value, dict):
        return False
    required = set(ModelSpec.__required_keys__)
    if not required.issubset(value.keys()):
        return False
    return all(isinstance(key, str) and key.isidentifier() for key in value)


def is_serialized_model_spec(value: object, /) -> TypeIs[SerializedModelSpec]:
    if not is_model_spec(value):
        return False
    required = set(SerializedModelSpec.__required_keys__)
    return required.issubset(value.keys())


def spec_from_tensor(tensor: Tensor, path: FilePath) -> SerializedTensorSpec:
    return {
        "__tensor__": "torch",
        "path": str(path),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }


def import_tensor(spec: SerializedTensorSpec) -> Tensor:
    tensor = torch.load(spec["path"])
    validate_tensor_spec(tensor, spec)
    return tensor


def deserialize_model(spec: SerializedModelSpec) -> nn.Module:
    fmt = spec["__format__"]
    path = spec["__state_dict__"]
    if path is None:
        raise ValueError("Serialized model spec missing '__state_dict__'.")

    match fmt:
        case "torchscript":
            module = torch.jit.load(path)
            assert isinstance(module, nn.Module)
            return module
        case "torch_export":
            imported = torch.export.load(path)
            return imported.module()
        case "state_dict":
            instance = initialize_model_from_spec(spec)
            state = torch.load(path)
            instance.load_state_dict(state)
            return instance
        case _:
            raise ValueError(f"Unsupported model format: {fmt!r}")


def initialize_model[T: nn.Module](arg: Makes[T], /) -> T:
    spec = infer_spec(arg)
    return initialize_model_from_spec(spec)


def initialize_model_from_spec(spec: ModelSpec) -> nn.Module:
    module_name = spec["__module__"]
    class_name = spec["__name__"]

    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    params = {key: value for key, value in spec.items() if not is_dunder(key)}

    if issubclass(cls, SupportsFromConfig):
        return cls.from_config(params)
    return cls(**params)


def validate_tensor_spec(tensor: Tensor, spec: SerializedTensorSpec) -> None:
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


def validate_model_spec(model: nn.Module | type[nn.Module], spec: ModelSpec) -> None:
    if not is_model_spec(spec):
        raise TypeError("Invalid model spec.")
    if any(not key.isidentifier() for key in spec):
        raise TypeError("Model spec keys must be identifiers.")

    model_cls = model if isinstance(model, type) else model.__class__
    assert issubclass(model_cls, nn.Module)

    expected_module = spec["__module__"]
    expected_name = spec["__name__"]

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

    if isinstance(model, nn.Module) and supports_config(model):
        hp_keys = set(model.hyperparameters.keys())
        spec_keys = {key for key in spec if not is_dunder(key)}
        missing = spec_keys - hp_keys
        if missing:
            raise ValueError(f"Model spec keys not present in HP: {sorted(missing)}")


def spec_from_dataclass(config: object) -> ModelSpec:
    assert is_dataclass(config) and not isinstance(config, type)
    model_name = config.__class__.__qualname__.split(".", 1)[0]
    model_spec: ModelSpec = {
        "__module__": config.__class__.__module__,
        "__name__": model_name,
        "__spec_version__": SPEC_VERSION,
        "__module_version__": _infer_module_version(config.__class__.__module__),
    } | asdict(config)
    return model_spec


def infer_spec[T: nn.Module](spec: Makes[T]) -> ModelSpec:
    match spec:
        case nn.Module():
            model_spec = infer_spec(spec)
        case dict():
            model_spec = dict(spec)
        case type() as cls:
            model_spec = {"__module__": cls.__module__, "__name__": cls.__qualname__}
        case dtc if is_dataclass(dtc):
            model_spec = spec_from_dataclass(dtc)
        case _:
            raise TypeError(f"Unsupported model spec type: {type(spec).__name__}")

    if not is_model_spec(model_spec):
        raise TypeError("Expected a model spec dictionary.")

    module_name = model_spec["__module__"]
    class_name = model_spec["__name__"]
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    validate_model_spec(cls, model_spec)
    return model_spec


def validate_config(conf: object) -> None:
    match conf:
        case _ if is_scalar(conf):
            pass
        case Tensor():
            pass
        case nn.Module():
            pass
        case list() | tuple():
            for item in conf:
                validate_config(item)
        case dict():
            if (
                is_tensor_spec(conf)
                or is_model_spec(conf)
                or is_serialized_model_spec(conf)
            ):
                return
            for key, value in conf.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Config dict keys must be str, got {type(key).__name__}"
                    )
                if not key.isidentifier():
                    raise TypeError(f"Config dict key must be identifier: {key!r}")
                validate_config(value)
        case _:
            raise TypeError(f"Unsupported config type: {type(conf).__name__}")


def export_tensor(tensor: Tensor, path: FilePath) -> SerializedTensorSpec:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    return {
        "__tensor__": "torch",
        "path": str(path),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }


def export_model(
    arg: nn.Module | ExportedProgram, path: FilePath
) -> SerializedModelSpec:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = infer_spec(arg)

    match arg:
        case torch.jit.RecursiveScriptModule():
            torch.jit.save(arg, str(path))
            fmt = "torchscript"
        case ExportedProgram():
            torch.export.save(arg, str(path))
            fmt = "torch_export"
        case nn.Module():
            torch.save(arg.state_dict(), path)
            fmt = "state_dict"
        case _:
            raise TypeError(f"Expected nn.Module, got {type(arg).__name__}")

    serialized_spec: SerializedModelSpec = spec | {
        "__state_dict__": str(path),
        "__format__": fmt,
    }
    return serialized_spec


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


def export_config(arg: object, path: FilePath) -> None:
    validate_config(arg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = path.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    def convert(value: object, prefix: str) -> object:
        match value:
            case _ if (
                is_scalar(value)
                or is_tensor_spec(value)
                or is_model_spec(value)
                or is_serialized_model_spec(value)
            ):
                return value
            case Tensor():
                return export_tensor(value, assets_dir / f"{prefix}.pt")
            case ExportedProgram():
                return export_model(value, assets_dir / f"{prefix}.pt")
            case nn.Module():
                return export_model(value, assets_dir / f"{prefix}.pt")
            case list():
                return [convert(item, f"{prefix}_{i}") for i, item in enumerate(value)]
            case tuple():
                return [convert(item, f"{prefix}_{i}") for i, item in enumerate(value)]
            case dict():
                return {k: convert(v, f"{prefix}_{k}") for k, v in value.items()}
            case _:
                raise TypeError(f"Unsupported config type: {type(value).__name__}")

    payload = convert(arg, "config")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


class CustomReLU(nn.Module):
    @dataclass
    class DEFAULT_CONFIG:
        inplace: bool = False

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu_(x) if self.inplace else torch.relu(x)


class ReverseDense(nn.Module):
    @dataclass
    class DEFAULT_CONFIG:
        input_size: int
        output_size: int
        bias: bool = True
        activation: Makes[Activation] = CustomReLU

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        activation: str | nn.Module | dict[str, Any],
    ) -> None:
        super().__init__()

        cfg = self.DEFAULT_CONFIG(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            activation=activation,
        )
        self.HP = infer_spec(cfg)

        self.linear = nn.Linear(input_size, output_size, bias=bias)
        self.activation = initialize_model(activation)


def test_initialization() -> None:
    cfg = ReverseDense.DEFAULT_CONFIG(input_size=4, output_size=8)
    model = ReverseDense(**asdict(cfg))

    assert isinstance(model.activation, CustomReLU)
    assert model.linear.in_features == 4
    assert model.linear.out_features == 8


def test_export() -> None:
    cfg = ReverseDense.DEFAULT_CONFIG(input_size=4, output_size=8)
    export_config(asdict(cfg), Path("demo_config.json"))
