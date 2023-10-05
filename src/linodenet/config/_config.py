r"""LinODE-Net Configuration."""

__all__ = [
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
from importlib import import_module, resources
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import torch
import yaml

__logger__ = logging.getLogger(__name__)


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


def generate_folders(d: dict, /, *, current_path: Path) -> None:
    r"""Create nested folder structure based on nested dictionary index.

    References:
        - https://stackoverflow.com/a/22058144/9318372
    """
    for directory in d:
        path = current_path.joinpath(directory)
        if d[directory] is None:
            __logger__.debug("creating folder %s", path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            generate_folders(d[directory], current_path=path)


class ConfigMeta(type):
    """Metaclass for Config."""

    def __init__(
        cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwds: Any
    ) -> None:
        super().__init__(name, bases, namespace, **kwds)

        if "LOGGER" not in namespace:
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")


class Config(metaclass=ConfigMeta):
    r"""Configuration Interface."""

    LOGGER: ClassVar[logging.Logger]
    r"""Logger for the class."""

    DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r"""The default `torch` device to use."""
    DEFAULT_DTYPE = torch.float32
    r"""The default `torch` datatype to use."""
    _autojit: bool = True

    def __init__(self):
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
        assert isinstance(value, bool)
        self._autojit = bool(value)
        os.environ["LINODENET_AUTOJIT"] = str(value)

    @cached_property
    def CONFIG_FILE(self) -> dict:
        r"""Return dictionary containing basic configuration of TSDM."""
        path = resources.files(__package__) / "config.yaml"
        with path.open("r", encoding="utf8") as file:
            # with open(file, "r", encoding="utf8") as f:
            return yaml.safe_load(file)


class Project:
    """Holds Project related data."""

    DOC_URL = "https://bvt-htbd.gitlab-pages.tu-berlin.de/kiwi/tf1/linodenet/"

    @cached_property
    def NAME(self) -> str:
        r"""Get project name."""
        return self.ROOT_PACKAGE.__name__

    @cached_property
    def ROOT_PACKAGE(self) -> ModuleType:
        r"""Get project root package."""
        hierarchy = __package__.split(".")
        return import_module(hierarchy[0])

    @cached_property
    def ROOT_PATH(self) -> Path:
        r"""Return the root directory."""
        assert len(self.ROOT_PACKAGE.__path__) == 1
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
            """Results directory."""

            TEST_RESULTS_PATH = self.TEST_RESULTS_PATH

            def __setitem__(self, key, value):
                raise RuntimeError("ResultsDir is read-only!")

            def __getitem__(self, key):
                if key not in self:
                    path = self.TEST_RESULTS_PATH / Path(key).stem
                    path.mkdir(parents=True, exist_ok=True)
                    super().__setitem__(key, path)
                return super().__getitem__(key)

        return ResultsDir()

    def make_test_folders(self, dry_run: bool = True) -> None:
        r"""Make the tests folder if it does not exist."""
        package_structure = get_package_structure(self.ROOT_PACKAGE)

        def flattened(d: dict[str, Any], /) -> list[str]:
            r"""Flatten nested dictionary."""
            return list(d) + list(chain.from_iterable(map(flattened, d.values())))

        for package in flattened(package_structure):
            test_package_path = self.TESTS_PATH / package.replace(".", "/")
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


# logging.basicConfig(
#     filename=str(LOGDIR.joinpath("example.log")),
#     filemode="w",
#     format="[%(asctime)s] [%(levelname)-s]\t[%(name)s]\t%(message)s, (%(filename)s:%(lineno)s)",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     level=logging.INFO)
