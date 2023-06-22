#!/usr/bin/env python
"""Prints the direct dependencies of a module line by line.."""

import ast
import importlib
import pkgutil
import sys
from contextlib import redirect_stderr, redirect_stdout


def extract_imports(node: ast.Import) -> set[str]:
    """Visit an `import` statement node and extract third-party dependencies."""
    dependencies = set()
    for alias in node.names:
        module = alias.name.split(".")[0]
        if not module.startswith("_"):
            dependencies.add(module)
    return dependencies


def extract_import_from(node: ast.ImportFrom) -> set[str]:
    """Visit an `import from` statement node and extract third-party dependencies."""
    dependencies = set()
    module = node.module.split(".")[0]
    if not module.startswith("_"):
        dependencies.add(module)
    return dependencies


def is_submodule(submodule_name: str, module_name: str) -> bool:
    """True if submodule_name is a submodule of module_name."""
    return submodule_name.startswith(module_name + ".")


def get_dependencies(module_name: str, recursive: bool = True) -> set[str]:
    """Retrieve the list of third-party dependencies imported by a module."""
    module = importlib.import_module(module_name)
    dependencies = set()

    # Visit the current module
    with open(module.__file__, "r") as file:
        tree = ast.parse(file.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies |= extract_imports(node)
            elif isinstance(node, ast.ImportFrom):
                dependencies |= extract_import_from(node)

    if not recursive:
        return dependencies

    # if it is a package, recurse into it.

    if hasattr(module, "__path__"):
        # Visit the sub-packages/modules of the package
        for module_info in pkgutil.walk_packages(module.__path__):
            submodule = importlib.import_module(module_name + "." + module_info.name)
            submodule_name = submodule.__name__

            if is_submodule(submodule_name, module_name):
                dependencies |= get_dependencies(submodule_name, recursive=recursive)

    return dependencies


def group_dependencies(dependencies: set[str]) -> tuple[list[str], list[str]]:
    """Splits the dependencies into first-party and third-party."""
    stdlib_dependencies = set()
    third_party_dependencies = set()

    for dependency in dependencies:
        if dependency in sys.stdlib_module_names:
            stdlib_dependencies.add(dependency)
        else:
            third_party_dependencies.add(dependency)

    return sorted(stdlib_dependencies), sorted(third_party_dependencies)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python get_dependencies.py [module_name]")
        sys.exit(1)

    module_name = sys.argv[1]

    with redirect_stdout(None), redirect_stderr(None):
        dependencies = get_dependencies(module_name)

    _, third_party_dependencies = group_dependencies(dependencies)

    for dependency in third_party_dependencies:
        print(dependency)


if __name__ == "__main__":
    main()
