# AGENTS.md

Project conventions for automated agents contributing to `linodenet`.

## Project snapshot

- Language: Python
- Packaging: `pyproject.toml` (PEP 621)
- Target runtime: Python `>=3.14`, `<3.15` on Linux

## Code Style

- Prefer `match` over long `if`/`elif` chains when it improves clarity.
- Avoid deep nesting; favor early returns when sensible.
- Use f-strings for formatting.
- Prefer comprehensions over `for`-loops and `map`/`filter`.
- Use context managers for resources.

## Typing

- Type checking: `pyright`, `mypy`.
- Use builtin generics and PEP 604 unions (`list[int]`, `T | U`).
- Use `Optional[T]` only for optional kwargs and return types; otherwise `T | None`.
- Prefer abstract types for inputs (`Sequence`, `Mapping`) and concrete types for outputs.

## Function signatures

- Use positional-only (`/`) when parameter names add no semantic value.
- Keep positional-or-keyword params to at most two; prefer keyword-only (`*`).

## Tooling and docs

- Follow formatting and linting configured in `pyproject.toml`.
- Docstrings follow Google style; document invariants and edge cases.
