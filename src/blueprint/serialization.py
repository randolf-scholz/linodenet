r"""Archive serialization for PyTorch modules and exported programs."""
# ruff: file-ignore[SIM103]

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
from torch.jit import RecursiveScriptModule
from typing_extensions import TypedDict

from .core import (
    JSON,
    Blueprint,
    blueprint_to_json,
    infer_blueprint,
    initialize,
    is_blueprint,
)
from .torch import is_model_blueprint

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
    r"""JSON-serializable storage metadata for a saved model.

    Store the archive path, payload format, schema version, and module version.
    The archive written by `serialize_model` has the following layout:

    ```text
    archive.zip
    ├─ blueprint.json         Storage metadata
    ├─ hyperparameters.json   Inferred or provided hyperparameters
    ├─ model.pt               Serialized payload
    ├─ model_init.json        Constructor blueprint, for state-dict storage only
    └─ requirements.txt       Environment snapshot, when available
    ```
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
    r"""Storage metadata and archive layout for a module's state dictionary.

    ```text
    archive.zip
    ├─ blueprint.json         Storage metadata
    ├─ hyperparameters.json   Inferred or provided hyperparameters
    ├─ model_init.json        Constructor blueprint
    ├─ model.pt               PyTorch state dictionary
    └─ requirements.txt       Environment snapshot, when available
    ```
    """

    __storage_format__: ReadOnly[Literal["state_dict"]]
    __blueprint__: ReadOnly[Blueprint[T]]
    __assets__: ReadOnly[list[str]]


class SavedTorchScriptBlueprint[T: RecursiveScriptModule](
    SavedModelBlueprint[T],
    TypedDict,
):
    r"""Storage metadata and archive layout for a TorchScript module.

    ```text
    archive.zip
    ├─ blueprint.json         Storage metadata
    ├─ hyperparameters.json   Inferred or provided hyperparameters
    ├─ model.pt               TorchScript payload
    └─ requirements.txt       Environment snapshot, when available
    ```
    """

    __storage_format__: ReadOnly[Literal["torchscript"]]


class SavedTorchExportBlueprint[T: ExportedProgram](
    SavedModelBlueprint[T],
    TypedDict,
):
    r"""Storage metadata and archive layout for a PyTorch exported program.

    ```text
    archive.zip
    ├─ blueprint.json         Storage metadata
    ├─ hyperparameters.json   Inferred or provided hyperparameters
    ├─ model.pt               Exported program payload
    └─ requirements.txt       Environment snapshot, when available
    ```
    """

    __storage_format__: ReadOnly[Literal["torch_export"]]


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
    r"""Write installed package names and versions to an archive entry.

    Each line has the form `name==version`. If collecting the environment fails
    or yields no packages, no entry is written; collection failures are logged.
    """
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
    r"""Write a model archive and return its storage blueprint.

    Args:
        model: A module, TorchScript module, or exported program. Ordinary modules
            are saved as a state dictionary with a separate constructor blueprint.
        filepath: Archive path ending in `.pt` or `.zip`. Parent directories are
            created, and an existing file is overwritten.
        hyperparameters: Optional keyword arguments to record as metadata. If
            omitted, infer them from the model; failed inference is logged and
            produces empty hyperparameter metadata.

    Returns:
        Storage metadata containing the archive path, payload format, schema
        version, and module version. State-dict storage also requires successful
        constructor-blueprint inference, independently of hyperparameter metadata.

    Raises:
        ValueError: If the file extension is unsupported.
        TypeError: If the model or supplied hyperparameters have an unsupported type.
    """
    # ensure path ends with .pt or .zip
    path = Path(filepath)
    if path.suffix not in {".pt", ".zip"}:
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
            hp = {"__args__": [], "__kwargs__": dict(kwargs)}
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
    r"""Read an archive's storage blueprint and deserialize its model.

    The blueprint's `__storage_path__` determines the archive opened for payload
    loading. Both paths must identify existing files ending in `.pt` or `.zip`.

    Returns:
        The reconstructed module, including an executable module for export storage.

    Raises:
        FileNotFoundError: If an archive path does not identify an existing file.
        ValueError: If a file extension or payload format is unsupported.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    if path.suffix not in {".pt", ".zip"}:
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
    r"""Load a model from the archive identified by a storage blueprint.

    State-dict storage initializes a module from `model_init.json` and loads its
    saved state. TorchScript storage loads a scripted module. Export storage loads
    an exported program and returns its executable module.

    Raises:
        TypeError: If `spec` is not a serialized model blueprint.
        FileNotFoundError: If `__storage_path__` does not identify an existing file.
        ValueError: If the file extension or storage format is unsupported.
    """
    if not is_serialized_model_blueprint(spec):
        raise TypeError("Expected a serialized model spec dictionary.")

    path = Path(spec["__storage_path__"])

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    if path.suffix not in {".pt", ".zip"}:
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
