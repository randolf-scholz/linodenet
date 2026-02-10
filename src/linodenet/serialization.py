r"""Functions for serializing and deserializing PyTorch models and tensors."""

__all__ = [
    "SavedModelBluePrint",
    "SavedTensorBluePrint",
    "is_serialized_model_blueprint",
    # functions
    "serialize_model",
    "deserialize_model",
    "deserialize_model_from_blueprint",
    "serialize_tensor",
    "deserialize_tensor",
    "deserialize_tensor_from_blueprint",
]

import json
import os
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import ReadOnly, TypedDict, TypeIs
from zipfile import ZipFile

import torch
from torch import Tensor, nn
from torch.export import ExportedProgram

from linodenet.config import (
    ArgKey,
    ArgValue,
    BluePrint,
    FilePath,
    _infer_object_blueprint,
    _validate_tensor_blueprint,
    initialize,
    is_blueprint,
)

FORMAT_VERSION = "1.0"
r"""CONST: The version of the model spec format."""

type FilePath = str | os.PathLike[str]
# The Types allowed in serialized specs. (no actual models or tensors, only specs)
type ArgKey = str  # identifier, except dunder or sunder

# type JSON_Leaf = None | bool | int | float | str
# type JSON_Value = JSON_Leaf | list[str] | dict[str, JSON_Value]
# type JSON = dict[str, JSON_Value]

type JSON_Leaf = None | bool | int | float | str
type JSON_List[T: JSON_Value = JSON_Value] = list[T]
type JSON_Dict[T: JSON_Value = JSON_Value] = dict[str, T]
type JSON_Value[T: JSON_Value = JSON_Value] = JSON_Leaf | JSON_List[T] | JSON_Dict[T]
type JSON[T: JSON_Value] = dict[str, T]


class SavedTensorBluePrint(BluePrint[Tensor], TypedDict):
    __storage_path__: ReadOnly[str]
    __storage_format__: ReadOnly[str]  # e.g. "torch", "numpy", "tf", "safetensors"
    __spec_version__: ReadOnly[str]
    __module_name__: ReadOnly[str]  # e.g. "torch", "numpy", "tf"
    __module_version__: ReadOnly[str]
    r"""Version of the library that created the tensor, e.g. torch.__version__"""


class SavedModelBluePrint[T: nn.Module | ExportedProgram](BluePrint[T], TypedDict):
    r"""In memory representation of a JSON-serializable model specification."""

    # __module_name__: ReadOnly[str]
    # __class_name__: ReadOnly[str]
    #
    # __args__: ReadOnly[list[JSON_Value]]
    # __kwargs__: ReadOnly[dict[ArgKey, JSON_Value]]

    # new keys for serialized models
    __spec_version__: ReadOnly[str]
    __module_version__: ReadOnly[str]
    __storage_path__: ReadOnly[str]
    __storage_format__: ReadOnly[str]
    # **DunderKey: object (reserved for future use)


def is_serialized_model_blueprint(arg: object, /) -> TypeIs[SavedModelBluePrint]:
    if not is_blueprint(arg):
        return False
    if not SavedModelBluePrint.__required_keys__.issubset(arg.keys()):
        return False
    if not isinstance(arg.get("__storage_format__"), str):
        return False
    if not isinstance(arg.get("__storage_path__"), str):
        return False
    return True


def _infer_module_version(arg: str | ModuleType, /) -> str | None:
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


def _config_to_blueprint(value: ArgValue, *, verify_init: bool = True) -> JSON_Value:
    match value:
        case nn.Module():
            return _infer_object_blueprint(value, verify_init=verify_init)
        case list():
            return [_config_to_blueprint(item) for item in value]
        case tuple():
            return tuple(_config_to_blueprint(item) for item in value)
        case dict():
            return {key: _config_to_blueprint(item) for key, item in value.items()}
        case _:
            return value


def serialize_model[M: nn.Module | ExportedProgram](
    model: M, filepath: FilePath, /
) -> SavedModelBluePrint[M]:
    spec = _infer_object_blueprint(model)
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
            "__spec_version__": FORMAT_VERSION,
            "__module_version__": _infer_module_version(model.__class__.__module__),
        }
        archive.writestr("config.json", json.dumps(spec))

    return spec


def deserialize_model(path: FilePath, /) -> nn.Module:
    r"""Deserialize a model from a file."""
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

    assert is_serialized_model_blueprint(spec)
    return deserialize_model_from_blueprint(spec)


def deserialize_model_from_blueprint[M: nn.Module | ExportedProgram](
    spec: SavedModelBluePrint[M], /
) -> M:
    r"""Deserialize a model from a blueprint spec."""
    if not is_serialized_model_blueprint(spec):
        raise TypeError("Expected a serialized model spec dictionary.")

    path = Path(spec["__storage_path__"])

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    if path.suffix not in (".pt", ".zip"):
        raise ValueError("Model file extension must be .pt or .zip")

    with (
        ZipFile(path, "r") as archive,
        archive.open("model.pt", "r") as model_file,
    ):
        match fmt := spec["__storage_format__"]:
            case "torchscript":
                module = torch.jit.load(model_file)
                assert isinstance(module, nn.Module)
                return module
            case "torch_export":
                imported = torch.export.load(model_file)
                return imported.module()
            case "torch_state_dict" | "state_dict":
                with archive.open("blueprint.json", "r") as blueprint_file:
                    conf = json.load(blueprint_file)
                instance = initialize(conf)
                state = torch.load(model_file)
                instance.load_state_dict(state)
                return instance
            case _:
                raise ValueError(f"Unsupported model format: {fmt!r}")


def serialize_tensor(tensor: Tensor, path: FilePath, /) -> SavedTensorBluePrint:
    r"""Serialize a tensor to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    match path.suffix:
        case ".pt":
            fmt = "torch"
            torch.save(tensor, path)
        case ".safetensors":
            fmt = "safetensors"
            import safetensors.torch

            safetensors.torch.save_file(tensor, path)
        case _:
            raise NotImplementedError(f"Unsupported tensor format: {path.suffix!r}")

    return {
        "__storage_path__": str(path),
        "__storage_format__": fmt,
        "__spec_version__": FORMAT_VERSION,
        "__module_name__": tensor.__class__.__module__,
        "__module_version__": _infer_module_version(tensor.__class__.__module__),
    }


def deserialize_tensor(path: FilePath, /) -> Tensor:
    r"""Deserialize a tensor from a file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tensor file not found: {path}")

    match path.suffix:
        case ".pt":
            return torch.load(path)
        case ".safetensors":
            import safetensors.torch

            return safetensors.torch.load_file(path)
        case _:
            raise NotImplementedError(f"Unsupported tensor format: {path.suffix!r}")


def deserialize_tensor_from_blueprint(spec: SavedTensorBluePrint, /) -> Tensor:
    r"""Deserialize a tensor from a blueprint spec."""
    tensor = deserialize_tensor(spec["__storage_path__"])
    _validate_tensor_blueprint(tensor, spec)
    return tensor
