r"""Base protocols and ordering utilities for domain definitions."""

__all__ = ["Domain", "PosetEnum"]


from collections.abc import Mapping
from enum import Enum
from functools import cache
from typing import ClassVar, Protocol, Self

from torch import Tensor


class Domain(Protocol):
    r"""Protocol for Domains."""

    def __contains__(self, item: Tensor, /) -> Tensor: ...

    def __le__(self, other: Self, /) -> bool: ...

    def __lt__(self, other: Self, /) -> bool: ...

    def __or__(self, other: Self, /) -> Self: ...

    def __and__(self, other: Self, /) -> Self: ...


class PosetEnum(Enum):
    r"""Mixin implementing a partial order from immediate-superset edges."""

    KNOWN_EDGES: ClassVar[Mapping[Self, frozenset[Self]]]
    r"""Dependencies"""
    KNOWN_TAGS: ClassVar[Mapping[Self, frozenset[Self]]]
    r"""Reverse dependencies."""

    @classmethod
    def _top_node(cls) -> Self | None:
        for name in ("TOP", "ANY"):
            if name in cls.__members__:
                return cls.__members__[name]
        return None

    @classmethod
    def _bottom_node(cls) -> Self | None:
        for name in ("BOTTOM", "NONE"):
            if name in cls.__members__:
                return cls.__members__[name]
        return None

    @classmethod
    @cache
    def _compiled_edges(cls) -> Mapping[Self, frozenset[Self]]:
        edges: dict[Self, set[Self]] = {node: set() for node in cls}

        for src, targets in cls.KNOWN_EDGES.items():  # type: ignore[attr-defined]
            edges[src].update(targets)

        for tag, members in cls.KNOWN_TAGS.items():  # type: ignore[attr-defined]
            for member in members:
                edges[member].add(tag)

        if (top := cls._top_node()) is not None:
            for node in cls:
                if node is not top:
                    edges[node].add(top)

        if (bottom := cls._bottom_node()) is not None:
            edges[bottom].update(node for node in cls if node is not bottom)

        return {src: frozenset(targets) for src, targets in edges.items()}

    @classmethod
    @cache
    def _validated_edges(cls) -> Mapping[Self, frozenset[Self]]:
        edges: Mapping[Self, frozenset[Self]] = cls._compiled_edges()
        members = frozenset(cls)

        if bad_keys := {node for node in edges if node not in members}:
            raise TypeError(f"Expected {cls.__name__} nodes, got {bad_keys!r}.")

        all_targets = frozenset().union(*edges.values())
        if bad_targets := {target for target in all_targets if target not in members}:
            raise TypeError(f"Expected {cls.__name__} targets, got {bad_targets!r}.")

        stack: set[Self] = set()
        visited: set[Self] = set()

        def visit(node: Self, /) -> None:
            if node in stack:
                raise ValueError(f"Cycle detected in {cls.__name__} order at {node!r}.")
            if node in visited:
                return

            stack.add(node)
            for target in edges.get(node, frozenset()):
                visit(target)
            stack.remove(node)
            visited.add(node)

        for node in cls:
            visit(node)

        return edges

    @classmethod
    @cache
    def _upward_closure(cls, node: Self, /) -> frozenset[Self]:
        edges = cls._validated_edges()
        parents = edges.get(node, frozenset())

        closure = frozenset({node, *parents})
        for parent in parents:
            closure = closure | cls._upward_closure(parent)

        return closure

    def __le__(self, other: object, /) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return other in type(self)._upward_closure(self)

    def __lt__(self, other: object, /) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self <= other and self != other

    def __str__(self) -> str:
        return str(self.value)
