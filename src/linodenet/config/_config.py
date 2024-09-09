r"""LinODE-Net Configuration."""
# ruff: noqa: N802

__all__ = [
    # Constants
    "CONFIG",
    "PROJECT",
    # Classes
    "Config",
    "Project",
    # Functions
    "generate_folders",
    "get_package_structure",
]

import logging
import os
from functools import cached_property
from importlib import import_module
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Final


def get_package_structure(root_module: ModuleType, /) -> dict[str, Any]:
    r"""Creates nested dictionary of the package structure."""
    d = {}
    for name in dir(root_module):
        attr = getattr(root_module, name)
        # check if it is a subpackage
        if (
            isinstance(attr, ModuleType)
            and attr.__name__.startswith(root_module.__name__)
            and attr.__package__ != root_module.__package__
            and attr.__package__ is not None
        ):
            d[attr.__package__] = get_package_structure(attr)
    return d


def generate_folders(dirs: str | list | dict, /, *, parent: Path) -> None:
    r"""Create nested folder structure based on nested dictionary index.

    References:
        https://stackoverflow.com/a/22058144/9318372
    """
    match dirs:
        case str(name):
            path = parent.joinpath(name)
            path.mkdir(parents=True, exist_ok=True)
        case list(items):
            for item in items:
                generate_folders(item, parent=parent)
        case dict(mapping):
            for key, value in mapping.items():
                generate_folders(value, parent=parent.joinpath(key))
        case _:
            raise TypeError


class Config:
    r"""Configuration Interface."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    r"""Logger for the class."""

    _autojit: bool = True

    def __init__(self) -> None:
        r"""Initialize the configuration."""
        # TODO: Should be initialized by an init/toml file.
        os.environ["LINODENET_AUTOJIT"] = "True"
        self._autojit: bool = True

    @property
    def autojit(self) -> bool:
        r"""Whether to automatically jit-compile the models."""
        return self._autojit

    @autojit.setter
    def autojit(self, value: bool) -> None:
        self._autojit = bool(value)
        os.environ["LINODENET_AUTOJIT"] = str(value)


class Project:
    r"""Holds Project related data."""

    DOC_URL = "https://bvt-htbd.gitlab-pages.tu-berlin.de/kiwi/tf1/linodenet/"

    @cached_property
    def NAME(self) -> str:
        r"""Get project name."""
        return self.ROOT_PACKAGE.__name__

    @cached_property
    def ROOT_PACKAGE(self) -> ModuleType:
        r"""Get project root package."""
        if __package__ is None:
            raise ValueError(f"Unexpected package: {__package__=}")
        hierarchy = __package__.split(".")
        return import_module(hierarchy[0])

    @cached_property
    def ROOT_PATH(self) -> Path:
        r"""Return the root directory."""
        if len(self.ROOT_PACKAGE.__path__) != 1:
            raise ValueError(f"Unexpected path: {self.ROOT_PACKAGE.__path__=}")

        path = Path(self.ROOT_PACKAGE.__path__[0])

        if path.parent.stem != "src":
            raise ValueError(
                f"This seems to be an installed version of {self.NAME},"
                f" as {path} is not in src/*"
            )
        return path.parent.parent

    @cached_property
    def DOCS_PATH(self) -> Path:
        r"""Return the `docs` directory."""
        docs_path = self.ROOT_PATH / "docs"
        if not docs_path.exists():
            raise ValueError(f"Docs directory {docs_path} does not exist!")
        return docs_path

    @cached_property
    def SOURCE_PATH(self) -> Path:
        r"""Return the source directory."""
        source_path = self.ROOT_PATH / "src"
        if not source_path.exists():
            raise ValueError(f"Source directory {source_path} does not exist!")
        return source_path

    @cached_property
    def TESTS_PATH(self) -> Path:
        r"""Return the test directory."""
        tests_path = self.ROOT_PATH / "tests"
        if not tests_path.exists():
            raise ValueError(f"Tests directory {tests_path} does not exist!")
        return tests_path

    @cached_property
    def TEST_RESULTS_PATH(self) -> Path:
        r"""Return the test `results` directory."""
        return self.TESTS_PATH / "results"

    @cached_property
    def RESULTS_DIR(self) -> dict[str | Path, Path]:
        r"""Return the `results` directory."""

        class ResultsDir(dict):
            r"""Results directory."""

            TEST_RESULTS_PATH = self.TEST_RESULTS_PATH

            def __setitem__(self, key: str | Path, value: Path, /) -> None:
                raise RuntimeError("ResultsDir is read-only!")

            def __getitem__(self, key: str | Path, /) -> Path:
                if key not in self:
                    path = self.TEST_RESULTS_PATH / Path(key).stem
                    path.mkdir(parents=True, exist_ok=True)
                    super().__setitem__(key, path)
                return super().__getitem__(key)

        return ResultsDir()

    def make_test_folders(self, *, dry_run: bool = True) -> None:
        r"""Make the tests folder if it does not exist."""
        package_structure = get_package_structure(self.ROOT_PACKAGE)

        def flattened(d: dict[str, Any], /) -> list[str]:
            r"""Flatten nested dictionary."""
            return list(d) + list(chain.from_iterable(map(flattened, d.values())))

        for package in flattened(package_structure):
            test_package_path = self.TESTS_PATH / package.replace(".", "/")
            test_package_init_file = test_package_path / "__init__.py"

            if not test_package_path.exists():
                if dry_run:
                    print(f"Dry-Run: Creating {test_package_path}")
                else:
                    print(f"Creating {test_package_path}")
                    test_package_path.mkdir(parents=True, exist_ok=True)
            if not test_package_path.exists():
                if dry_run:
                    print(f"Dry-Run: Creating {test_package_init_file}")
                else:
                    raise RuntimeError(f"Creation of {test_package_path} failed!")
            elif not test_package_init_file.exists():
                if dry_run:
                    print(f"Dry-Run: Creating {test_package_init_file}")
                else:
                    print(f"Creating {test_package_init_file}")
                    message = f'"""Tests for {package}."""\n'
                    test_package_init_file.write_text(message, encoding="utf8")
        if dry_run:
            print("Pass option `dry_run=False` to actually create the folders.")


# region CONSTANTS
PROJECT: Final[Project] = Project()
r"""Project configuration."""

CONFIG: Final[Config] = Config()
r"""Configuration Class."""
# endregion CONSTANTS
