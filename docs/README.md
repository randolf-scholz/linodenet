# Building documentation

Run these commands from the repository root:

- `just docs::build` (or `just docs::make`) builds the Zensical site in
  `docs/build/site` with strict validation.
- `just docs::build --clean` clears the Zensical cache before building.
- `just docs::clean` removes `docs/build`, including staged sources and the site.

From this directory, omit the `docs::` prefix. The build commands accept extra
arguments for Zensical.

Authored pages and their attachments live in `content/`. The build runs
`generate_reference.py`, which copies that directory recursively into `build/sources/`,
stages static files, and generates API reference Markdown and `reference/SUMMARY.md`
navigation from the Python source. It reads the staging path from `docs_dir` in
`zensical.toml`. Each run replaces the staged sources so renamed or deleted pages
do not linger. Zensical renders those sources into `build/site/` and reuses its
cache during normal builds.

The home, contributing, and license pages are relative symlinks in `content/`:
`index.md` → `../../README.md`, `CONTRIBUTING.md` → `../../CONTRIBUTING.md`,
and `LICENSE.md` → `../../LICENSE`. Edit the root files to update the site.
Staging follows these symlinks and copies their contents without rewriting links.
References to repository-only files in shared pages use filenames in backticks.

The generator creates the entire reference section, including the landing page
at `build/sources/reference/index.md` with its heading and package list.

Browser files live in `static/`: stylesheets in `static/css/` and MathJax
configuration in `static/mathjax_config.js`. The build copies the entire
directory into `build/sources/static/`, preserving its layout. Local static paths
in `zensical.toml` also exist relative to this directory before building.
HTML templates live separately in `templates/`. Zensical loads them directly;
they are not copied into the site sources.

Put images and attachments used by repository-root Markdown files in the root
`assets/` directory. The relative symlink `content/assets` → `../../assets` makes
links such as `assets/lssm.png` work both in the repository README and on the site.
Keep documentation-only attachments near their pages under `content/`, for
example in `content/diagrams/` or `content/theory/`.

This README contains maintainer instructions and is not copied into the site.

The generated site uses explicit `.html` links so pages can be opened directly
from disk. The default web fonts load from Google Fonts, with system fonts as an
offline fallback. MathJax's runtime, math fonts, and extensions load from a CDN,
so rendering mathematics requires an internet connection unless those resources
are already cached. External reference links still lead to their respective
websites.
