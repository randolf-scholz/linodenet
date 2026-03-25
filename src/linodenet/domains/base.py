r"""Base protocols and ordering utilities for domain definitions."""

__all__ = [
    "Domain",
    "Intersection",
    "Union",
    "Inverse",
    "PosetEnum",
]


from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from typing import ClassVar, Protocol, Self

from torch import Tensor


class Domain(Protocol):
    r"""Protocol for Domains."""

    def __contains__(self, item: Tensor, /) -> bool:
        raise NotImplementedError

    def __le__(self, other: Self, /) -> bool:
        return NotImplemented

    def __lt__(self, other: Self, /) -> bool:
        return NotImplemented

    def __gt__(self, other: Self, /) -> bool:
        return NotImplemented

    def __ge__(self, other: Self, /) -> bool:
        return NotImplemented

    def __invert__(self) -> Domain:
        return Inverse(self)

    def __or__(self, other: Self, /) -> Domain:
        return Union({self, other})

    def __and__(self, other: Self, /) -> Domain:
        return Intersection({self, other})


@dataclass(frozen=True)
class Union[D: Domain](Domain):
    r"""Structural union of matrix domains."""

    domains: frozenset[D]

    def __init__(self, domains: Iterable[D] = (), /) -> None:
        object.__setattr__(self, "domains", frozenset(domains))

    def __contains__(self, item: Tensor, /) -> bool:
        return any(item in domain for domain in self.domains)

    def __iter__(self) -> Iterator[D]:
        return iter(self.domains)

    def __len__(self) -> int:
        return len(self.domains)


@dataclass(frozen=True)
class Intersection[D: Domain](Domain):
    r"""Structural intersection of matrix domains."""

    domains: frozenset[D] = field(default_factory=frozenset)

    def __init__(self, domains: Iterable[D] = (), /) -> None:
        object.__setattr__(self, "domains", frozenset(domains))

    def __contains__(self, item: Tensor, /) -> bool:
        return all(item in domain for domain in self.domains)

    def __iter__(self) -> Iterator[D]:
        return iter(self.domains)

    def __len__(self) -> int:
        return len(self.domains)


@dataclass(frozen=True)
class Inverse[D: Domain](Domain):
    r"""Structural complement of a domain."""

    domain: D

    def __contains__(self, item: Tensor, /) -> bool:
        return item not in self.domain


class PosetEnum(Enum):
    r"""Mixin implementing a partial order from immediate-superset edges."""

    KNOWN_EDGES: ClassVar[Mapping[Self, frozenset[Self]]]  # pyright: ignore[reportInvalidTypeForm]
    r"""Dependencies"""
    KNOWN_TAGS: ClassVar[Mapping[Self, frozenset[Self]]]  # pyright: ignore[reportInvalidTypeForm]
    r"""Reverse dependencies."""
    KNOWN_MEETS: ClassVar[Sequence[tuple[Self, frozenset[Self]]]]  # pyright: ignore[reportInvalidTypeForm]
    r"""Named meet rules encoded as implications x≤aᵢ ∀i ⇒ x≤m."""

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

        for meet, factors in getattr(cls, "KNOWN_MEETS", ()):
            edges[meet].update(factors)

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
    def _validated_meets(cls) -> tuple[tuple[Self, frozenset[Self]], ...]:
        meets = tuple(getattr(cls, "KNOWN_MEETS", ()))
        members = frozenset(cls)

        if bad_keys := {node for node, _ in meets if node not in members}:
            raise TypeError(f"Expected {cls.__name__} meet nodes, got {bad_keys!r}.")

        all_factors = frozenset().union(*(factors for _, factors in meets))
        if bad_factors := {factor for factor in all_factors if factor not in members}:
            raise TypeError(
                f"Expected {cls.__name__} meet factors, got {bad_factors!r}."
            )

        if empty_meets := {node for node, factors in meets if not factors}:
            raise ValueError(f"Expected non-empty meet factors, got {empty_meets!r}.")

        return meets

    @classmethod
    @cache
    def _upward_closure(cls, node: Self, /) -> frozenset[Self]:
        edges = cls._validated_edges()
        meets = cls._validated_meets()

        closure: set[Self] = set()
        stack = [node]

        while stack:
            current = stack.pop()
            if current in closure:
                continue

            closure.add(current)
            stack.extend(edges.get(current, frozenset()))

            for meet, factors in meets:
                if meet not in closure and factors <= closure:
                    stack.append(meet)

        return frozenset(closure)

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
