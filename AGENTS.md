# AGENTS.md

Project conventions for automated agents contributing to `linodenet`.

## Project snapshot

- Language: Python
- Packaging: `pyproject.toml` (PEP 621)
- Target runtime: Python `>=3.14`, `<3.15` on Linux

## Code Style

- Prefer `match` over long `if`/`elif` chains when it improves clarity.
- Avoid deep nesting; refactor only when it materially improves clarity.
- Prefer straightforward inline code over extracting tiny helper functions.
- Introduce a helper function only when it is reused, materially improves readability, or isolates non-trivial logic
  that would otherwise obscure the main control flow.
- As a rule of thumb, only extract a helper when it leads to a meaningful reduction in code at the call sites; if the
  diff is roughly line-neutral, keep the logic inline.
- Do not extract one- or two-line private helpers for single-use expressions unless they encode an important domain
  concept.
- Keep related logic in the main method when the extracted helper would force the reader to jump around the file to
  understand a short sequence of operations.
- Avoid function definitions inside functions.
- Exception: local function definitions are acceptable for decorators or when a callback/closure is clearly the most
  readable option.
- Use f-strings for formatting.
- Prefer comprehensions over `for`-loops and `map`/`filter`.
- Avoid nested loops inside comprehensions; use an explicit loop when flattening or combining iterables.
- Use relative imports only when importing from sibling submodules; use absolute imports otherwise.
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
- Run shell commands with an explicit timeout; use `30s` by default unless a different limit is justified.
  For example, prefer `timeout 30s <command>`.
- For project-local Python commands, set `PYTHONPATH=src` by default unless there is a specific reason not to.
  This applies to `python`, `pytest`, `pyright`, `mypy`, ad-hoc scripts, and one-off import checks.
- Docstrings follow Google style; document invariants and edge cases.
- Prefer raw docstrings (`r"""..."""`) by default.
- Use Unicode characters in latex formulas in docstrings to improve readability.
  so $ℝ$ rather than $\mathbb{R}$, $ϕ$ rather that $\phi$, $∑$ rather than $\sum$, etc.
  This includes super- and subscripts, e.g. $xᵢⱼ$ rather than $x_{ij}$ and $f⁻¹$ rather than $f^{-1}$.
- In chat responses, prefer plain Unicode math and symbols over LaTeX/MathJax markup because this console does not
  render MathJax.
- We use `sphinx-dollarmath`, so prefer `$...$` inline math over `:math:` for better readability.
  Use `.. math::` for display math.
- Matplotlib supports CPU torch tensors directly; prefer passing CPU tensors rather than converting to NumPy.
- Prefer f-string debug syntax (`{var=}`) for simple diagnostic prints.
- `torch.jit` is deprecated for this project; do not introduce `torch.jit` usage in new code.

## running tests

- export `PYTHONPATH=src` and disable the `pytest-rerunfailures` plugin and skip benchmark tests:
  `PYTHONPATH=src pytest -p no:rerunfailures --benchmark-skip <tests>`.
