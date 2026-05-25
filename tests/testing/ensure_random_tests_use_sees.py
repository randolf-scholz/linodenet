"""AST-based checker for randomized test-case builders.

This checker enforces a simple convention for test helpers:
all functions and methods whose name starts with ``make_`` must define a
mandatory keyword-only ``rng`` argument without a default value.
"""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

DEFAULT_ROOTS: Final[tuple[Path, ...]] = (Path("tests"),)
PYTHON_SUFFIX: Final[str] = ".py"


@dataclass(frozen=True)
class Violation:
    r"""A lint violation reported by this checker."""

    path: Path
    line: int
    column: int
    name: str
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.name}: {self.message}"


class MakeRngChecker(ast.NodeVisitor):
    r"""Collect violations for ``make_*`` functions without ``*, rng``."""

    def __init__(self, path: Path, /) -> None:
        self.path = path
        self.violations: list[Violation] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check(node)
        self.generic_visit(node)

    def _check(self, node: ast.FunctionDef | ast.AsyncFunctionDef, /) -> None:
        if not node.name.startswith("make_"):
            return

        kwonlyargs = node.args.kwonlyargs
        kwdefaults = node.args.kw_defaults
        for argument, default in zip(kwonlyargs, kwdefaults, strict=True):
            if argument.arg != "rng":
                continue
            if default is None:
                return
            self._report(
                node,
                "must define a mandatory keyword-only 'rng' argument without a default",
            )
            return

        self._report(
            node,
            "must define a mandatory keyword-only 'rng' argument without a default",
        )

    def _report(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, message: str, /
    ) -> None:
        self.violations.append(
            Violation(
                path=self.path,
                line=node.lineno,
                column=node.col_offset + 1,
                name=node.name,
                message=message,
            )
        )


def iter_python_files(path: Path, /) -> list[Path]:
    r"""Return Python files rooted at ``path``."""
    if path.is_file():
        return [path] if path.suffix == PYTHON_SUFFIX else []
    return sorted(file for file in path.rglob(f"*{PYTHON_SUFFIX}") if file.is_file())


def check_file(path: Path, /) -> list[Violation]:
    r"""Parse ``path`` and return all rng-related violations."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    checker = MakeRngChecker(path)
    checker.visit(tree)
    return checker.violations


def main(argv: list[str] | None = None) -> int:
    r"""Run the checker and return a shell exit code."""
    args = argv if argv is not None else sys.argv[1:]
    roots = [Path(arg) for arg in args] if args else list(DEFAULT_ROOTS)

    violations: list[Violation] = []
    for root in roots:
        for path in iter_python_files(root):
            violations.extend(check_file(path))

    if not violations:
        return 0

    for violation in violations:
        print(violation.format())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
