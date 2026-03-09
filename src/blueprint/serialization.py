r"""Functions for serializing and deserializing PyTorch models and tensors."""
# ruff: noqa: SIM103

__all__ = [
    # CONSTANTS
    "CONFIG",
    # Types
    "SavedModelBlueprint",
    "SavedStateDictBlueprint",
    "SavedTorchScriptBlueprint",
    "SavedTorchExportBlueprint",
    # "SavedTensorBlueprint",
    "is_serialized_model_blueprint",
    # functions
    "serialize_model",
    "deserialize_model",
    "deserialize_model_from_blueprint",
    "write_environment",
]

import json
import logging
import os
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import IO, Literal, Optional, ReadOnly, TypeGuard, cast
from zipfile import ZipFile

import torch
from torch import nn
from torch.export import ExportedProgram
from torch.jit import (  # type: ignore[attr-defined]
    RecursiveScriptModule,  # pyright: ignore[reportPrivateImportUsage]
)
from typing_extensions import TypedDict

from blueprint.core import (
    JSON,
    Blueprint,
    blueprint_to_json,
    infer_blueprint,
    initialize,
    is_blueprint,
)
from blueprint.torch import is_model_blueprint

__logger__ = logging.getLogger(__name__)


class CONFIG:
    r"""Configuration constants for model serialization."""

    FORMAT_VERSION = "1.0"
    MODEL_FILE = "model.pt"
    HYPERPARAMETERS_FILE = "hyperparameters.json"
    BLUEPRINT_FILE = "blueprint.json"
    MODEL_INIT_FILE = "model_init.json"
    ENVIRONMENT_FILE = "requirements.txt"


type FilePath = str | os.PathLike[str]
type FileLike = FilePath | IO[bytes]
# The Types allowed in serialized specs. (no actual models or tensors, only specs)


# type JSON_Leaf = None | bool | int | float | str
# type JSON_List[T: JSON_Value = JSON_Value] = list[T]
# type JSON_Dict[T: JSON_Value = JSON_Value] = dict[str, T]
# type JSON_Value[T: JSON_Value = JSON_Value] = JSON_Leaf | JSON_List[T] | JSON_Dict[T]
# type JSON[T: JSON_Value] = dict[str, T]


class SavedModelBlueprint[T](TypedDict):
    r"""In memory representation of a JSON-serializable model specification.

    Storage schema (zip archive):
    archive.zip
    ├─ blueprint.json         (required) saved model blueprint, includes storage metadata
    ├─ hyperparameters.json   (required) inferred or provided hyperparameters
    ├─ model.<ext>            (required) serialized model file, e.g. .pt or .zip
    ├─ pylock.toml            (optional) environment lockfile
    ├─ requirements.txt       (optional) environment requirements
    └─ etc.
    """

    # new keys for serialized models
    __spec_version__: ReadOnly[str]
    __module_version__: ReadOnly[str]
    __storage_path__: ReadOnly[str]
    __storage_format__: ReadOnly[str]
    # **DunderKey: object (reserved for future use)


class SavedStateDictBlueprint[T: nn.Module](
    SavedModelBlueprint[T],
    TypedDict,
):
    r"""State-dict storage layout.

    archive.zip
    ├─ blueprint.json         (required) includes storage metadata
    ├─ hyperparameters.json   (required) inferred or provided hyperparameters
    ├─ model_init.json        (required) model blueprint for initialization
    ├─ model.pt               (required) torch state_dict
    └─ assets/                (optional) tensors referenced by model_init.json
    """

    __storage_format__: ReadOnly[Literal["state_dict"]]  # type: ignore[misc]
    __blueprint__: ReadOnly[Blueprint[T]]
    __assets__: ReadOnly[list[str]]


class SavedTorchScriptBlueprint[T: RecursiveScriptModule](
    SavedModelBlueprint[T],
    TypedDict,
):
    r"""TorchScript storage layout.

    archive.zip
    ├─ blueprint.json         (required) storage metadata only
    ├─ hyperparameters.json   (required) inferred or provided hyperparameters
    └─ model.pt               (required) torchscript payload
    """

    __storage_format__: ReadOnly[Literal["torchscript"]]  # type: ignore[misc]


class SavedTorchExportBlueprint[T: ExportedProgram](
    SavedModelBlueprint[T],
    TypedDict,
):
    r"""Torch export storage layout.

    archive.zip
    ├─ blueprint.json         (required) storage metadata only
    ├─ hyperparameters.json   (required) inferred or provided hyperparameters
    └─ model.pt               (required) torch export payload
    """

    __storage_format__: ReadOnly[Literal["torch_export"]]  # type: ignore[misc]


def is_serialized_model_blueprint(arg: object, /) -> TypeGuard[SavedModelBlueprint]:
    if not is_blueprint(arg):
        return False
    if not SavedModelBlueprint.__required_keys__.issubset(arg.keys()):
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
    except Exception as exc:
        __logger__.exception(exc)

    if isinstance(arg, ModuleType):
        module = arg
    else:
        try:
            module = __import__(arg)
        except Exception as exc:
            __logger__.exception(exc)
            return None

    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def _collect_environment_requirements() -> list[str]:
    requirements: list[str] = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        version = dist.version
        if not name or not version:
            continue
        requirements.append(f"{name}=={version}")
    return sorted(requirements, key=str.casefold)


def write_environment(archive: ZipFile, filename: str, /) -> None:
    r"""Write a simple requirements-style environment snapshot into the archive."""
    try:
        requirements = _collect_environment_requirements()
    except Exception as exc:
        __logger__.exception(exc)
        return
    if not requirements:
        return
    archive.writestr(filename, "\n".join(requirements) + "\n")


def serialize_model[M: nn.Module | ExportedProgram](
    model: M,
    filepath: FilePath,
    /,
    *,
    hyperparameters: Optional[JSON] = None,
) -> SavedModelBlueprint[M]:
    r"""Serialize a model to a file and return its blueprint spec."""
    # ensure path ends with .pt or .zip
    path = Path(filepath)
    if path.suffix not in (".pt", ".zip"):
        raise ValueError("Model file extension must be .pt or .zip")
    path.parent.mkdir(parents=True, exist_ok=True)

    # determine the hyperparameters to save
    hp: JSON
    match hyperparameters:
        case None:
            try:
                hp = blueprint_to_json(infer_blueprint(model))
            except Exception as exc:
                __logger__.exception(exc)
                hp = {}
        case {**kwargs}:
            hp = {"__args__": [], "__kwargs__": dict(kwargs)}  # type: ignore[arg-type]
        case _:
            raise TypeError(f"Unsupported type: {type(hyperparameters)}")

    # write model payload and metadata files
    with ZipFile(path, "w") as archive:
        archive.writestr(
            CONFIG.HYPERPARAMETERS_FILE,
            json.dumps(hp, indent="\t"),
        )
        write_environment(archive, CONFIG.ENVIRONMENT_FILE)
        with archive.open(CONFIG.MODEL_FILE, "w") as model_file:
            match model:
                case RecursiveScriptModule():
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
        if fmt == "state_dict":
            model_init: JSON = blueprint_to_json(infer_blueprint(model))
            assert is_model_blueprint(model_init)
            archive.writestr(
                CONFIG.MODEL_INIT_FILE,
                json.dumps(model_init, indent="\t"),
            )
        blueprint: JSON = {
            "__storage_path__": str(path),
            "__storage_format__": fmt,
            "__spec_version__": CONFIG.FORMAT_VERSION,
            "__module_version__": _infer_module_version(model.__class__.__module__),
        }
        archive.writestr(
            CONFIG.BLUEPRINT_FILE,
            json.dumps(blueprint, indent="\t"),
        )

    assert is_serialized_model_blueprint(blueprint)
    return blueprint


def deserialize_model(path: FilePath, /) -> nn.Module:
    r"""Deserialize a model from a file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    if path.suffix not in (".pt", ".zip"):
        raise ValueError("Model file extension must be .pt or .zip")

    with (
        ZipFile(path, "r") as archive,
        archive.open(CONFIG.BLUEPRINT_FILE, "r") as config_file,
    ):
        spec = json.load(config_file)

    assert is_serialized_model_blueprint(spec)
    return deserialize_model_from_blueprint(spec)


def deserialize_model_from_blueprint[M: nn.Module | ExportedProgram](
    spec: SavedModelBlueprint[M], /
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
                return cast("M", module)
            case "torch_export":
                imported = torch.export.load(model_file)
                return cast("M", imported.module())
            case "torch_state_dict" | "state_dict":
                with archive.open(CONFIG.MODEL_INIT_FILE, "r") as model_init_file:
                    model_init = json.load(model_init_file)
                instance: nn.Module = initialize(model_init)
                state = torch.load(model_file)
                instance.load_state_dict(state)
                return cast("M", instance)
            case _:
                raise ValueError(f"Unsupported model format: {fmt!r}")
