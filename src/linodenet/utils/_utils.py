r"""Utility functions."""

__all__ = [
    # Constants
    "PROJECT_ROOT",
    "PROJECT_TEST",
    "PROJECT_SOURCE",
    # Functions
    "autojit",
    "deep_dict_update",
    "deep_keyval_update",
    "flatten_nested_tensor",
    "get_package_structure",
    "get_project_root_package",
    "initialize_from",
    "initialize_from_config",
    "is_dunder",
    "make_test_folders",
    "pad",
    # Classes
]

import logging
import warnings
from collections.abc import Iterable, Mapping
from copy import deepcopy
from functools import partial, wraps
from importlib import import_module
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import Any, Final, TypeVar

import torch
from torch import Tensor, jit, nn

from linodenet.config import conf

__logger__ = logging.getLogger(__name__)

ObjectType = TypeVar("ObjectType")
r"""Generic type hint for instances."""

nnModuleType = TypeVar("nnModuleType", bound=nn.Module)
r"""Type Variable for nn.Modules."""


@jit.script
def pad(
    x: Tensor,
    value: float,
    pad_width: int,
    dim: int = -1,
    prepend: bool = False,
) -> Tensor:
    r"""Pad a tensor with a constant value along a given dimension."""
    shape = list(x.shape)
    shape[dim] = pad_width
    z = torch.full(shape, value, dtype=x.dtype, device=x.device)

    if prepend:
        return torch.cat((z, x), dim=dim)
    return torch.cat((x, z), dim=dim)


def deep_dict_update(d: dict, new: Mapping, inplace: bool = False) -> dict:
    r"""Update nested dictionary recursively in-place with new dictionary.

    References
    ----------
    - https://stackoverflow.com/a/30655448/9318372
    """
    if not inplace:
        d = deepcopy(d)

    for key, value in new.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_dict_update(d.get(key, {}), value)
        else:
            d[key] = new[key]
    return d


def deep_keyval_update(d: dict, **new_kv: Any) -> dict:
    r"""Update nested dictionary recursively in-place with key-value pairs.

    References
    ----------
    - https://stackoverflow.com/a/30655448/9318372
    """
    for key, value in d.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_keyval_update(d.get(key, {}), **new_kv)
        elif key in new_kv:
            d[key] = new_kv[key]
    return d


def autojit(base_class: type[nnModuleType]) -> type[nnModuleType]:
    r"""Class decorator that enables automatic jitting of nn.Modules upon instantiation.

    Makes it so that

    .. code-block:: python

        class MyModule:
            ...


        model = jit.script(MyModule())

    and

    .. code-block:: python


        class MyModule:
            ...


        model = MyModule()

    are (roughly?) equivalent
    """
    assert issubclass(base_class, nn.Module)

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type: ignore[misc, valid-type]
        r"""A simple Wrapper."""

        # noinspection PyArgumentList
        def __new__(cls, *args: Any, **kwargs: Any) -> nnModuleType:  # type: ignore[misc]
            # Note: If __new__() does not return an instance of cls,
            # then the new instance's __init__() method will not be invoked.
            instance: nnModuleType = base_class(*args, **kwargs)

            if conf.autojit:
                scripted: nnModuleType = jit.script(instance)
                return scripted
            return instance

    assert issubclass(WrappedClass, base_class)
    return WrappedClass


def flatten_nested_tensor(inputs: Tensor | Iterable[Tensor]) -> Tensor:
    r"""Flattens element of general Hilbert space."""
    if isinstance(inputs, Tensor):
        return torch.flatten(inputs)
    if isinstance(inputs, Iterable):
        return torch.cat([flatten_nested_tensor(x) for x in inputs])
    raise ValueError(f"{inputs=} not understood")


def initialize_from(
    lookup_table: ModuleType | dict[str, type[ObjectType]],
    /,
    __name__: str,
    **kwargs: Any,
) -> ObjectType:
    r"""Lookup class/function from dictionary and initialize it.

    Roughly equivalent to:

    .. code-block:: python

        obj = lookup_table[__name__]
        if isclass(obj):
            return obj(**kwargs)
        return partial(obj, **kwargs)
    """
    warnings.warn("Use initialize_from_config instead", DeprecationWarning)
    if isinstance(lookup_table, ModuleType):
        assert hasattr(lookup_table, __name__)
        obj: type[ObjectType] = getattr(lookup_table, __name__)
    else:
        obj = lookup_table[__name__]

    assert callable(obj), f"Looked up object {obj} not callable class/function."

    # check that obj is a class, but not metaclass or instance.
    if isinstance(obj, type) and not issubclass(obj, type):
        return obj(**kwargs)
    # if it is function, fix kwargs
    return partial(obj, **kwargs)  # type: ignore[return-value]


def initialize_from_config(config: dict[str, Any]) -> nn.Module:
    r"""Initialize a class from a dictionary."""
    assert "__name__" in config, "__name__ not found in dict"
    assert "__module__" in config, "__module__ not found in dict"
    __logger__.debug("Initializing %s", config)
    config = config.copy()
    module = import_module(config.pop("__module__"))
    cls = getattr(module, config.pop("__name__"))
    opts = {key: val for key, val in config.items() if not is_dunder("key")}
    obj = cls(**opts)
    assert isinstance(obj, nn.Module)
    return obj


def is_dunder(s: str, /) -> bool:
    r"""Check if name is a dunder method."""
    return s.isidentifier() and s.startswith("__") and s.endswith("__")


def get_project_root_package() -> ModuleType:
    r"""Get project root package."""
    return import_module(__package__.split(".", maxsplit=1)[0])


def _get_project_root_directory() -> Path:
    r"""Get project root directory."""
    root_package = get_project_root_package()
    assert len(root_package.__path__) == 1
    root_path = Path(root_package.__path__[0])
    assert root_path.parent.stem == "src", f"{root_path=} must be in src/*"
    return root_path.parent.parent


PROJECT_ROOT: Final[Path] = _get_project_root_directory()
"""Project root directory."""

PROJECT_TEST: Final[Path] = PROJECT_ROOT / "tests"
"""Project test directory."""

PROJECT_SOURCE: Final[Path] = PROJECT_ROOT / "src"
"""Project source directory."""


def get_package_structure(root_module: ModuleType, /) -> dict[str, Any]:
    r"""Create nested dictionary of the package structure."""
    d = {}
    for name in dir(root_module):
        attr = getattr(root_module, name)
        if isinstance(attr, ModuleType):
            # check if it is a subpackage
            if (
                attr.__name__.startswith(root_module.__name__)
                and attr.__package__ != root_module.__package__
                and attr.__package__ is not None
            ):
                d[attr.__package__] = get_package_structure(attr)
    return d


def make_test_folders(dry_run: bool = True) -> None:
    r"""Make the tests folder if it does not exist."""
    root_package = get_project_root_package()
    package_structure = get_package_structure(root_package)

    def _flatten(d: dict[str, Any], /) -> list[str]:
        r"""Flatten nested dictionary."""
        return list(d) + list(chain.from_iterable(map(_flatten, d.values())))

    packages: list[str] = _flatten(package_structure)
    for package in packages:
        test_package_path = PROJECT_TEST / package.replace(".", "/")
        test_package_init = test_package_path / "__init__.py"

        if not test_package_path.exists():
            if dry_run:
                print(f"Dry-Run: Creating {test_package_path}")
            else:
                print("Creating {test_package_path}")
                test_package_path.mkdir(parents=True, exist_ok=True)
        if not test_package_path.exists():
            if dry_run:
                print(f"Dry-Run: Creating {test_package_init}")
            else:
                raise RuntimeError(f"Creation of {test_package_path} failed!")
        elif not test_package_init.exists():
            if dry_run:
                print(f"Dry-Run: Creating {test_package_init}")
            else:
                print(f"Creating {test_package_init}")
                with open(test_package_init, "w", encoding="utf8") as file:
                    file.write(f'"""Tests for {package}."""\n')

    if dry_run:
        print("Pass option `dry_run=False` to actually create the folders.")
