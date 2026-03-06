r"""LinODE-Net Configuration."""
# ruff: noqa: N802

__all__ = [
    # Constants
    "PROJECT",
    # Classes
    "Project",
    # Functions
    "generate_folders",
    "get_package_structure",
]

import os
import tomllib
from functools import cached_property
from importlib import import_module
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import Any, Final

type PathLike = str | os.PathLike[str]


def _discover_root_packages(source_path: Path, /) -> list[ModuleType]:
    if not source_path.exists():
        raise ValueError(f"Source directory {source_path} does not exist!")
    candidates = [
        entry
        for entry in source_path.iterdir()
        if entry.is_dir() and (entry / "__init__.py").is_file()
    ]
    packages: list[ModuleType] = []
    errors: dict[str, Exception] = {}
    for candidate in candidates:
        try:
            pkg = import_module(candidate.name)
        except ModuleNotFoundError as exc:
            errors[candidate.name] = exc
        else:
            packages.append(pkg)
    if errors:
        raise ExceptionGroup(
            f"Failed to import root packages under {source_path}",
            list(errors.values()),
        )
    if not packages:
        raise ValueError(
            f"No root packages found under {source_path} (no candidates found)."
        )
    return packages


def _flattened_package_structure(d: dict[str, Any], /) -> list[str]:
    r"""Flatten nested dictionary."""
    return list(d) + list(
        chain.from_iterable(map(_flattened_package_structure, d.values()))
    )


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


class _DirGetter:
    r"""Results directory."""

    def __init__(self, base_dir: PathLike, /) -> None:
        self.base_dir = Path(base_dir)
        self.paths: dict[PathLike, Path] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(paths={self.paths})"

    def __getitem__(self, key: PathLike, /) -> Path:
        if key not in self.paths:
            path = self.base_dir / Path(key).stem
            path.mkdir(parents=True, exist_ok=True)
            self.paths[key] = path
        return self.paths[key]


class Project:
    r"""Holds Project related data."""

    @cached_property
    def ROOT_PATH(self) -> Path:
        r"""Return the root directory."""
        start = Path(__file__).resolve().parent
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate
        raise FileNotFoundError(
            f"Could not locate project root from {start}; pyproject.toml not found."
        )

    @cached_property
    def DOCS_PATH(self) -> Path:
        r"""Return the `docs` directory."""
        docs_path = self.ROOT_PATH / "docs"
        if not docs_path.is_dir():
            raise FileNotFoundError(f"Docs directory {docs_path} does not exist!")
        return docs_path

    @cached_property
    def SOURCE_PATH(self) -> Path:
        r"""Return the source directory."""
        source_path = self.ROOT_PATH / "src"
        if not source_path.is_dir():
            raise FileNotFoundError(f"Source directory {source_path} does not exist!")
        return source_path

    @cached_property
    def TESTS_PATH(self) -> Path:
        r"""Return the test directory."""
        tests_path = self.ROOT_PATH / "tests"
        if not tests_path.is_dir():
            raise FileNotFoundError(f"Tests directory {tests_path} does not exist!")
        return tests_path

    @cached_property
    def PROJECT_FILE(self) -> dict[str, Any]:
        r"""Return `pyproject.toml` as a dictionary."""
        project_file = self.ROOT_PATH / "pyproject.toml"
        if not project_file.is_file():
            raise FileNotFoundError(f"Project file {project_file} does not exist!")
        with project_file.open("rb") as handle:
            return tomllib.load(handle)

    @cached_property
    def NAME(self) -> str:
        r"""Get project name."""
        project = self.PROJECT_FILE.get("project")
        match project:
            case {"name": str(name)}:
                return name
            case _:
                raise ValueError("Missing `project.name` in pyproject.toml.")

    @cached_property
    def ROOT_PACKAGES(self) -> list[ModuleType]:
        r"""Get project root packages under `src`."""
        return _discover_root_packages(self.SOURCE_PATH)

    @cached_property
    def TEST_RESULTS_PATH(self) -> Path:
        r"""Return the test `results` directory."""
        return self.TESTS_PATH / ".results"

    @cached_property
    def RESULTS_DIR(self) -> _DirGetter:
        r"""Return the `results` directory."""
        return _DirGetter(self.TEST_RESULTS_PATH)

    def make_test_folders(self, *, dry_run: bool = True) -> None:
        r"""Make the tests folder if it does not exist."""
        packages = [
            get_package_structure(root_package) for root_package in self.ROOT_PACKAGES
        ]

        for dirs in chain.from_iterable(map(_flattened_package_structure, packages)):
            test_package_path = self.TESTS_PATH / dirs.replace(".", "/")
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
                    message = f'"""Tests for {dirs}."""\n'
                    test_package_init_file.write_text(message, encoding="utf8")
        if dry_run:
            print("Pass option `dry_run=False` to actually create the folders.")


PROJECT: Final[Project] = Project()
r"""Project configuration."""
