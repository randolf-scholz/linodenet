# AGENTS.md

Project conventions for automated agents contributing to `linodenet`.

## Project snapshot

- Language: Python
- Packaging: `pyproject.toml` (PEP 621)
- Target runtime: Python `>=3.14`, `<3.15` on Linux

## Code Style

- Prefer `match` over long `if`/`elif` chains when it improves clarity.
- Avoid deep nesting; refactor only when it materially improves clarity.
- Prefer straightforward inline code over extracting helper functions.
- Helpers that are only used once should usually be inlined.
- One- or two-line private helpers should usually be inlined.
- As a rule of thumb, prefer the version with fewer total lines of code overall.
- Introduce a helper function only when it is reused, materially improves readability, isolates non-trivial logic that
  would otherwise obscure the main control flow, or clearly reduces the total line count at the call sites.
- If extracting a helper is roughly line-neutral, keep the logic inline.
- Keep related logic in the main method when the extracted helper would force the reader to jump around the file to
  understand a short sequence of operations.
- Exception: a small helper is still acceptable when it encodes an important domain concept more clearly than inline
  code would.
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
- Run shell commands with an explicit timeout; use `30s` by default unless a different limit is justified. For example,
  prefer `timeout 30s <command>`.
- Use the repo-local virtual environment at `.venv` for all project-local Python commands. Prefer
  `.venv/bin/python -m <tool>` so the interpreter and installed packages are unambiguous.
- For project-local Python commands, set `PYTHONPATH=src` by default unless there is a specific reason not to. This
  applies to `python`, `pytest`, `pyright`, `mypy`, ad-hoc scripts, and one-off import checks.
- Docstrings follow Google style; document invariants and edge cases.
- Prefer raw docstrings (`r"""..."""`) by default.
- Use Unicode characters in latex formulas in docstrings to improve readability. so $ℝ$ rather than $\mathbb{R}$, $ϕ$
  rather that $\phi$, $∑$ rather than $\sum$, etc. This includes super- and subscripts, e.g. $xᵢⱼ$ rather than $x_{ij}$
  and $f⁻¹$ rather than $f^{-1}$.
- In chat responses, prefer plain Unicode math and symbols over LaTeX/MathJax markup because this console does not
  render MathJax.
- We use `sphinx-dollarmath`, so prefer `$...$` inline math over `:math:` for better readability. Use `.. math::` for
  display math.
- Matplotlib supports CPU torch tensors directly; prefer passing CPU tensors rather than converting to NumPy.
- Prefer f-string debug syntax (`{var=}`) for simple diagnostic prints.
- `torch.jit` is deprecated for this project; do not introduce `torch.jit` usage in new code.

## running tests

- disable the `pytest-rerunfailures` plugin and skip benchmark tests:
  `.venv/bin/python -m pytest -p no:rerunfailures --benchmark-skip <tests>`.

## Shell commands and approvals

- Run read-only inspection commands separately rather than combining them with
  `;`, `&&`, or other shell composition when there is no dependency between them.
- In particular, run `sed -n`, `rg -n`, `rg --files`, and read-only `git`
  commands such as `git diff`, `git status`, `git log`, and `git show`
  as individual commands.
- These commands are already permitted by the execution policy. Do not request elevated permissions or additional
  approval for them.
- Only combine commands when their execution genuinely depends on the previous command succeeding.
