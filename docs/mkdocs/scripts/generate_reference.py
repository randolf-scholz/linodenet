r"""Generate virtual MkDocs pages for public modules, classes, and functions."""

from __future__ import annotations

import ast
import posixpath
from dataclasses import dataclass
from pathlib import Path

import mkdocs_gen_files

ROOT_DIR = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT_DIR / "src"
REFERENCE_DIR = Path("reference")
NAV = mkdocs_gen_files.Nav()
DOCUMENTED_IDENTIFIERS: set[str] = set()


@dataclass(frozen=True)
class Module:
    r"""Source and public API metadata for one Python module."""

    identifier: str
    doc_path: Path
    exports: list[str]
    is_package: bool
    kinds: dict[str, str]
    nodes: list[ast.stmt]
    source_path: Path


def write_page(
    identifier: str,
    doc_path: Path,
    source_path: Path,
    nav_parts: tuple[str, ...],
    members: list[str] | None = None,
) -> None:
    r"""Create a virtual API page and add it to the generated navigation."""
    if identifier in DOCUMENTED_IDENTIFIERS:
        return

    DOCUMENTED_IDENTIFIERS.add(identifier)
    NAV[nav_parts] = doc_path.relative_to(REFERENCE_DIR).as_posix()
    with mkdocs_gen_files.open(doc_path, "w") as file:
        file.write(f"::: {identifier}\n")
        if members is not None:
            file.write("    options:\n      members:\n")
            file.writelines(f"        - {name!r}\n" for name in members)
    mkdocs_gen_files.set_edit_path(doc_path, source_path.relative_to(ROOT_DIR))


def get_exports(nodes: list[ast.stmt]) -> list[str]:
    r"""Return the names declared by a literal module ``__all__``."""
    for node in nodes:
        match node:
            case ast.Assign(targets=targets, value=value) if any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in targets
            ):
                try:
                    names = ast.literal_eval(value)
                except ValueError:
                    return []
                return [name for name in names if isinstance(name, str)]
    return []


def get_kinds(nodes: list[ast.stmt], exports: list[str]) -> dict[str, str]:
    r"""Classify directly declared public classes, functions, and constants."""
    kinds: dict[str, str] = {}
    for node in nodes:
        match node:
            case ast.ClassDef(name=name) if not name.startswith("_"):
                kinds[name] = "classes"
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) if not name.startswith("_"):
                kinds[name] = "functions"
            case ast.Assign(targets=targets):
                names = [target.id for target in targets if isinstance(target, ast.Name)]
                for name in names:
                    if not name.startswith("_") and (name in exports or name.isupper()):
                        kinds[name] = "constants"
            case ast.AnnAssign(target=ast.Name(id=name)) | ast.TypeAlias(name=ast.Name(id=name)):
                if not name.startswith("_") and (name in exports or name.isupper()):
                    kinds[name] = "constants"
    return kinds


def get_class_members(nodes: list[ast.stmt]) -> list[str]:
    r"""Return the public methods and attributes declared directly by a class."""
    members: list[str] = []
    for node in nodes:
        match node:
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name):
                if name == "__init__" or not name.startswith("_"):
                    members.append(name)
            case ast.Assign(targets=targets):
                members.extend(
                    target.id
                    for target in targets
                    if isinstance(target, ast.Name) and not target.id.startswith("_")
                )
            case ast.AnnAssign(target=ast.Name(id=name)) | ast.ClassDef(name=name):
                if not name.startswith("_"):
                    members.append(name)
    return members


MODULES: dict[str, Module] = {}
for source_path in sorted(SOURCE_DIR.rglob("*.py")):
    module_parts = source_path.relative_to(SOURCE_DIR).with_suffix("").parts
    is_package = module_parts[-1] == "__init__"
    match module_parts[-1]:
        case "__main__":
            continue
        case "__init__":
            module_parts = module_parts[:-1]

    if not module_parts or any(part.startswith("_") for part in module_parts):
        continue

    identifier = ".".join(module_parts)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    exports = get_exports(tree.body)
    MODULES[identifier] = Module(
        identifier=identifier,
        doc_path=REFERENCE_DIR.joinpath(*module_parts, "index.md"),
        exports=exports,
        is_package=is_package,
        kinds=get_kinds(tree.body, exports),
        nodes=tree.body,
        source_path=source_path,
    )


PAGE_PATHS = {
    f"{module.identifier}.{name}": module.doc_path.parent / name / "index.md"
    if kind == "classes"
    else module.doc_path.parent / f"{name}.md"
    for module in MODULES.values()
    for name, kind in module.kinds.items()
    if kind in {"classes", "functions"}
}


def get_imports(module: Module) -> dict[str, str]:
    r"""Return fully-qualified targets for names imported by a module."""
    package_parts = module.identifier.split(".")
    if not module.is_package:
        package_parts = package_parts[:-1]

    imports: dict[str, str] = {}
    for node in module.nodes:
        if not isinstance(node, ast.ImportFrom):
            continue
        target_parts = package_parts[: len(package_parts) - node.level + 1]
        if node.module:
            target_parts.extend(node.module.split("."))
        target_module = ".".join(target_parts)
        for alias in node.names:
            if alias.name != "*":
                imports[alias.asname or alias.name] = f"{target_module}.{alias.name}"
    return imports


def get_target(module: Module, name: str, seen: set[str] | None = None) -> str | None:
    r"""Resolve a directly declared or re-exported object to its canonical identifier."""
    identifier = f"{module.identifier}.{name}"
    if identifier in PAGE_PATHS:
        return identifier

    seen = set() if seen is None else seen
    if identifier in seen:
        return None
    seen.add(identifier)

    target = get_imports(module).get(name)
    if target is None:
        return None
    if target in PAGE_PATHS:
        return target

    target_module, _, target_name = target.rpartition(".")
    if target_module in MODULES:
        return get_target(MODULES[target_module], target_name, seen)
    return None


def get_kind(module: Module, name: str) -> str:
    r"""Return the documented category for a direct or re-exported module member."""
    if name in module.kinds:
        return module.kinds[name]

    target = get_target(module, name)
    if target is not None:
        target_module, _, target_name = target.rpartition(".")
        return MODULES[target_module].kinds[target_name]
    return "constants"


def get_relative_path(source: Path, target: Path) -> str:
    r"""Return a Markdown link from one generated document to another."""
    return posixpath.relpath(target.as_posix(), start=source.parent.as_posix())


def write_contents(module: Module, file: object) -> None:
    r"""Write the typed overview of a module's public API."""
    categories = {"classes": [], "functions": [], "submodules": [], "constants": []}
    names = module.exports or list(module.kinds)
    for name in names:
        kind = get_kind(module, name)
        target = get_target(module, name)
        if target is None:
            categories[kind].append(f"`{name}`")
        else:
            categories[kind].append(
                f"[{name}]({get_relative_path(module.doc_path, PAGE_PATHS[target])})"
            )

    prefix = f"{module.identifier}."
    for identifier, submodule in MODULES.items():
        if identifier.startswith(prefix) and identifier.count(".") == module.identifier.count(".") + 1:
            categories["submodules"].append(
                f"[{identifier.rsplit('.', 1)[-1]}]({get_relative_path(module.doc_path, submodule.doc_path)})"
            )

    if not any(categories.values()):
        return
    file.write("\n## Contents\n")
    for heading, entries in categories.items():
        if entries:
            file.write(f"\n### {heading.title()}\n\n")
            file.writelines(f"- {entry}\n" for entry in entries)


def document_module_definitions(module: Module) -> None:
    r"""Create pages for public module-level classes and functions."""
    positions = {name: index for index, name in enumerate(module.exports)}
    definitions = [
        node
        for node in module.nodes
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]
    for node in sorted(definitions, key=lambda node: positions.get(node.name, len(positions))):
        match node:
            case ast.ClassDef(name=name, body=body):
                write_page(
                    f"{module.identifier}.{name}",
                    PAGE_PATHS[f"{module.identifier}.{name}"],
                    module.source_path,
                    (*module.identifier.split("."), name),
                    get_class_members(body),
                )
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name):
                write_page(
                    f"{module.identifier}.{name}",
                    PAGE_PATHS[f"{module.identifier}.{name}"],
                    module.source_path,
                    (*module.identifier.split("."), name),
                )


for module in MODULES.values():
    write_page(
        module.identifier,
        module.doc_path,
        module.source_path,
        tuple(module.identifier.split(".")),
    )
    with mkdocs_gen_files.open(module.doc_path, "a") as file:
        write_contents(module, file)
        constants = [name for name, kind in module.kinds.items() if kind == "constants"]
        if constants:
            file.write("\n## Constants\n\n")
            file.write(f"::: {module.identifier}\n")
            file.write("    options:\n")
            file.write("      heading_level: 3\n")
            file.write("      show_docstring_description: false\n")
            file.write("      show_root_heading: false\n")
            file.write("      show_root_toc_entry: false\n")
            file.write("      members:\n")
            file.writelines(f"        - {name!r}\n" for name in constants)
    document_module_definitions(module)

with mkdocs_gen_files.open(REFERENCE_DIR / "SUMMARY.md", "w") as file:
    file.writelines(NAV.build_literate_nav())
