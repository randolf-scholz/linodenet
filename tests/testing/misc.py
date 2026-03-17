r"""Test testing."""

__all__ = [
    # Functions
    "camel2snake",
    "snake2camel",
]


def camel2snake(string: str) -> str:
    r"""Convert camel case to snake case."""
    return "".join(["_" + c.lower() if c.isupper() else c for c in string]).lstrip("_")


def snake2camel(string: str) -> str:
    r"""Convert snake case to camel case."""
    return "".join([c.title() for c in string.split("_")])
