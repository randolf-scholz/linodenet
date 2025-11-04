"""Allow `MathJax`_ to be used to display math in Sphinx's HTML writer.

This requires the MathJax JavaScript library on your webserver/computer.

.. _MathJax: https://www.mathjax.org/
"""

from __future__ import annotations

from types import NoneType
from typing import TYPE_CHECKING, Any


import sphinx
from sphinx.errors import ExtensionError
from pathlib import Path

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.util.typing import ExtensionMetadata

# more information for mathjax secure url is here:
# https://docs.mathjax.org/en/latest/web/start.html#using-mathjax-from-a-content-delivery-network-cdn
MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
MATHJAX4_URL = r"https://cdn.jsdelivr.net/npm/mathjax@4/tex-mml-chtml.js"


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_config_value(
        "custom_mathjax", None, "html", types=frozenset({str, NoneType})
    )

    def install_custom_mathjax(
        app: Sphinx,
        pagename: str,
        templatename: str,
        context: dict[str, Any],
        event_arg: Any,
    ):
        match app.config.custom_mathjax:
            case None:
                pass
            case str(config_filename):
                config_filepath = Path(config_filename)
                assert config_filepath.is_file(), "mathjax3_config file not found"
                assert config_filepath.suffix == ".js", (
                    "mathjax3_config must be a .js file"
                )
                with config_filepath.open(encoding="utf-8") as f:
                    body = f.read()
                app.add_js_file("", body=body)
            case _:
                raise ExtensionError("custom_mathjax must be a str (filename), or None")

    app.connect("html-page-context", install_custom_mathjax)

    return {
        "version": sphinx.__display_version__,
        "parallel_read_safe": True,
    }
