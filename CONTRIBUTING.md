# Contributing to imtskit

This guide describes the development workflow and conventions for `imtskit`.
Keep changes focused, add tests for new behavior and bug fixes, and update the
relevant documentation. Project configuration lives in [pyproject.toml](pyproject.toml);
[AGENTS.md](AGENTS.md) contains additional instructions for automated contributors.

## 1. Setting up the development environment

Development targets Linux and Python 3.14 (`>=3.14,<3.15`). Install `uv` for
environment and dependency management, and `just` for the repository's recipes.
The native extension requires a C++20 compiler; the default build also requires
the CUDA toolkit, cuDNN, and `clang-tidy`. CMake, Ninja, and the Python build tools
are included in the development dependencies.

Clone the repository (or your fork):

```bash
git clone https://github.com/randolf-scholz/imtskit.git
cd imtskit
```

Create the virtual environment, install the development dependencies, and build
the project in editable mode:

```bash
uv venv --python 3.14
source .venv/bin/activate
uv sync --group dev --no-install-project
uv sync --group dev
```

The first sync installs dependencies before building the project because its
build uses the environment's NumPy, PyTorch, and build tools without build
isolation. The second sync installs `imtskit` in editable mode, which is `uv`'s
default for this project. The `dev` group includes build, documentation, linting,
testing, and typing tools.

Python source edits are available immediately. Native C++ changes require
rebuilding the editable installation because automatic rebuilding is disabled:

```bash
uv sync --group dev --reinstall-package imtskit
```

Verify the installation and register the pre-commit hooks with `prek`:

```bash
uv run python -c "import imtskit; import imtskit_special"
uv run prek install
```

Keep the virtual environment activated when committing: some hooks invoke tools
from the active environment. Run all project-local Python commands through
`uv run <tool>`.

## 2. Documentation

Documentation is a work in progress. We are moving towards **Zensical and
Markdown**, replacing the Sphinx workflow. Write new documentation for the
Zensical site under [docs/zensical](docs/zensical); its configuration is
[docs/zensical.toml](docs/zensical.toml).

### Conventions

- Use Google-style docstrings, with sections such as `Args`, `Returns`, `Raises`,
  and `Examples` where useful.
- Prefer raw docstrings (`r"""..."""`).
- Document invariants, tensor shapes, assumptions, and edge cases so readers can
  understand the API's behavior.
- Use Markdown for new prose, including backticks for inline code and fenced
  code blocks with a language identifier for examples.
- Prefer readable Unicode symbols in mathematical formulas: `ℝ`, `ϕ`, `∑`,
  `xᵢⱼ`, and `f⁻¹`. Use `$...$` for inline math and `$$...$$` for display math
  in Markdown pages.

### Building the documentation

From the repository root, run:

```bash
just docs::zensical
```

This recipe regenerates the API reference Markdown from `src/`, builds the site
with strict validation, and copies static assets. The HTML output is written to
`docs/build/zensical/`; open `docs/build/zensical/index.html` to inspect it. API
pages under `docs/zensical/reference/` are generated, so update the source
docstrings or the generator instead of editing those pages directly.

The migration is incomplete: `just docs` still defaults to Sphinx. Use the
explicit `just docs::zensical` recipe for the new documentation workflow.

## 3. Linting & Formatting

Use `prek` to run the repository's configured checks. The hook configuration is
[prek.toml](prek.toml):

```bash
prek run -a  # prefix with `uv run` if venv not active
```

Hooks can modify files, including formatting and import ordering. Review the
changes, stage them, and rerun the checks before committing.

The main tools can also be run individually:

| Tool    | Purpose                                   | Command                     |
|---------|-------------------------------------------|-----------------------------|
| Pyrefly | Reference type checker                    | `pyrefly check src/ tests/` |
| Ruff    | Python linting and import checks          | `ruff check src/ tests/`    |
| Ruff    | Python formatting                         | `ruff format src/ tests/`   |
| Tach    | Module dependency and architecture checks | `tach check`                |

Pyrefly is the reference for typing decisions. Pyright is also configured and
runs through the hooks; it can be invoked with `uv run pyright`.

Ruff's rules and formatting settings are defined in `pyproject.toml`. To check
formatting without changing files, run `uv run ruff format --check src/ tests/`.
To apply lint fixes, run `uv run ruff check --fix src/ tests/` and review the result.

Tach checks imports against the module boundaries in [tach.toml](tach.toml), with
additional settings in `pyproject.toml`. Update declared dependencies when an
intentional architectural change adds imports across those boundaries.

## 4. Testing

`pytest` is used for testing. Invoke with

```bash
pytest  # prefix with `uv run` if venv not active
```

`pytest` is configured to discover tests in both `src/` and `tests/`, including doctests.

## 5. Type hinting

Use modern Python 3.14+ typing syntax, within the project's current Python 3.14
runtime constraint:

- Use builtin generics such as `list[int]`, `dict[str, Tensor]`, and `tuple[int, ...]`.
- Use PEP 604 unions (`T | U`) and `T | None`. Reserve `Optional[T]` for optional
  keyword arguments and return types; use `T | None` elsewhere.
- Use `type` statements for type aliases and type parameter syntax for generic
  functions and classes, such as `type Batch[T] = list[T]`.
- Use Python 3.14's lazy annotations. Do not add
  `from __future__ import annotations`.
- Prefer abstract input types such as `Sequence[T]` and `Mapping[K, V]`, and
  concrete output types such as `list[T]` and `dict[K, V]`.
- Strip longer module prefixes from annotations by importing the type directly:
  prefer `x: Tensor` over `x: torch.Tensor`. Short prefixes such as `nn.Module`
  are fine.
- Annotate function parameters and return values, and keep annotations consistent
  with the values accepted and returned at runtime.

For example:

```python
from collections.abc import Sequence

from torch import Tensor, nn

type TensorBatch = list[Tensor]


def apply_model(
        model: nn.Module, batches: Sequence[Tensor], /, *, copy: bool = False
) -> TensorBatch:
    outputs = [model(batch) for batch in batches]
    return [output.clone() for output in outputs] if copy else outputs
```

## 6. Code style

Follow the formatting and lint rules in `pyproject.toml`. Ruff formats Python
with four-space indentation, double quotes, and an 88-character target line
length.

### Control flow and helper functions

- Prefer `match`/`case` over long `if`/`elif` chains when it improves clarity.
- Avoid deep nesting; refactor when it materially improves readability.
- Prefer straightforward inline code. Helpers used only once, and one- or
  two-line private helpers, should usually be inlined.
- Introduce a helper when it is reused, materially improves readability, isolates
  non-trivial logic, or clearly reduces the total line count at call sites. A
  small helper can also be useful when it expresses an important domain concept.
- If extraction is roughly line-neutral, keep related logic in the main method
  so readers can follow a short sequence without jumping between functions.
  Prefer the version with fewer total lines when it remains clear.
- Avoid defining functions inside functions, except for decorators or when a
  callback or closure is clearly the most readable option.

### Expressions, imports, and resources

- Use f-strings for formatting and f-string debug syntax (`f"{value=}"`) for
  simple diagnostic prints.
- Prefer comprehensions over loops and `map`/`filter` when they remain readable.
  Avoid nested loops inside comprehensions; use an explicit loop for flattening
  or combining iterables.
- Use relative imports only for sibling submodules; use absolute imports otherwise.
- Use context managers for resources such as files.
- Pass CPU torch tensors directly to Matplotlib instead of converting them to
  NumPy solely for plotting.
- Do not introduce new `torch.jit` usage; it is deprecated for this project.

### Function signatures

- Use positional-only parameters (`/`) when parameter names add no semantic value.
- Keep positional-or-keyword parameters to at most two; prefer keyword-only
  parameters (`*`) for the rest.
