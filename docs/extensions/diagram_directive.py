r"""Extension for titled literal diagrams."""

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx


class Diagram(Directive):
    r"""Render a titled literal block in an admonition box.

    Usage::

        .. diagram:: Diagram

            x₁ ───▶ f₁(x₁)
            x₂ ───▶ f₂(x₂)
            ⋮
            xₙ ───▶ fₙ(xₙ)
    """

    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    has_content = True

    def run(self) -> list[nodes.Node]:
        r"""Create an admonition containing a literal diagram."""
        title_text = self.arguments[0] if self.arguments else "Diagram"
        text_nodes, messages = self.state.inline_text(title_text, self.lineno)
        diagram = nodes.admonition()
        diagram["classes"].append("diagram")
        diagram += nodes.title(title_text, "", *text_nodes)
        diagram += messages

        content = "\n".join(self.content)
        diagram += nodes.literal_block(content, content)
        return [diagram]


def setup(app: Sphinx) -> dict:
    r"""Install the extension."""
    app.add_directive("diagram", Diagram)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
