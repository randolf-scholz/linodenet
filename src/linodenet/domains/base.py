r"""Base protocols and ordering utilities for domain definitions."""

__all__ = [
    "Domain",
    "Intersection",
    "Meet",
    "Union",
    "Inverse",
    "PosetEnum",
]


from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from types import MappingProxyType
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


@dataclass(frozen=True)
class Meet:
    r"""Structural meet expression for poset labels."""

    factors: frozenset[PosetEnum]

    def __init__(self, factors: Iterable[PosetEnum] = (), /) -> None:
        nodes: set[PosetEnum] = set()
        for factor in factors:
            match factor:
                case Meet(factors=subfactors):
                    nodes.update(subfactors)
                case PosetEnum():
                    nodes.add(factor)
                case _:
                    raise TypeError(f"Expected PosetEnum factor, got {factor!r}.")
        object.__setattr__(self, "factors", frozenset(nodes))

    def __and__(self, other: PosetEnum | Meet, /) -> Meet:
        return Meet({*self.factors, other})

    def __iter__(self) -> Iterator[PosetEnum]:
        return iter(self.factors)

    def __len__(self) -> int:
        return len(self.factors)

    def __le__(self, other: object, /) -> bool:
        if not isinstance(other, PosetEnum):
            return NotImplemented
        types = {type(factor) for factor in self.factors}
        if len(types) != 1 or type(other) not in types:
            return NotImplemented
        cls = type(other)
        return other in cls._closure_from(self.factors)

    def __lt__(self, other: object, /) -> bool:
        if not isinstance(other, PosetEnum):
            return NotImplemented
        return self <= other and other not in self.factors

    def __ge__(self, other: object, /) -> bool:
        if not isinstance(other, PosetEnum):
            return NotImplemented
        return other <= self


class PosetEnum(Enum):
    r"""Mixin implementing a partial order from immediate-superset edges."""

    KNOWN_SUPERTYPES: ClassVar[Mapping[Self, frozenset[Self | Meet]]]
    r"""Dependencies"""
    KNOWN_SUBTYPES: ClassVar[Mapping[Self, frozenset[Self | Meet]]]
    r"""Reverse dependencies."""
    KNOWN_MEETS: ClassVar[Sequence[tuple[Self, Meet]]]
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
    def _parsed_supertypes(cls) -> Mapping[Self, frozenset[Self]]:
        r"""Parse declared supertypes into direct supertype relations.

        `KNOWN_SUPERTYPES` may contain plain nodes or meet expressions. A meet
        target `A & B` denotes the stronger statement `x ≤ A ∧ B`, so this
        parser expands it to the implied direct supertypes `A` and `B`.
        """
        raw_supertypes = cls.KNOWN_SUPERTYPES
        members = frozenset(cls)

        if bad_keys := {node for node in raw_supertypes if node not in members}:
            raise TypeError(f"Expected {cls.__name__} nodes, got {bad_keys!r}.")

        supertypes: dict[Self, frozenset[Self]] = {}
        for node, supers in raw_supertypes.items():
            if bad_targets := {
                target
                for target in supers
                if not isinstance(target, cls) and not isinstance(target, Meet)
            }:
                raise TypeError(
                    f"Expected {cls.__name__} or Meet targets, got {bad_targets!r}."
                )
            expanded_targets = frozenset(
                factor
                for target in supers
                for factor in (target if isinstance(target, Meet) else (target,))
            )
            if bad_targets := {
                target for target in expanded_targets if target not in members
            }:
                raise TypeError(
                    f"Expected {cls.__name__} targets, got {bad_targets!r}."
                )
            supertypes[node] = expanded_targets

        return supertypes

    @classmethod
    @cache
    def _compiled_supertypes(cls) -> Mapping[Self, frozenset[Self]]:
        r"""Compile all declared supertype information into one direct map.

        This combines parsed `KNOWN_SUPERTYPES`, reverse declarations from
        `KNOWN_SUBTYPES`, structural supertype relations induced by
        `KNOWN_MEETS`, and the implicit top/bottom node conventions.

        The result still contains only direct supertype edges. Transitive
        closure is computed separately by `_closure_from`.
        """
        supertypes: dict[Self, set[Self]] = {node: set() for node in cls}

        for node, supers in cls._parsed_supertypes().items():
            supertypes[node].update(supers)

        for meet, factors in cls._validated_meets():
            supertypes[meet].update(factors)

        for node, subtypes in cls._parsed_subtypes().items():
            for subtype in subtypes:
                supertypes[subtype].add(node)

        if (top := cls._top_node()) is not None:
            for node in cls:
                if node is not top:
                    supertypes[node].add(top)

        if (bottom := cls._bottom_node()) is not None:
            supertypes[bottom].update(node for node in cls if node is not bottom)

        return {src: frozenset(targets) for src, targets in supertypes.items()}

    @classmethod
    @cache
    def _parsed_subtypes(cls) -> Mapping[Self, frozenset[Self]]:
        r"""Parse declared subtypes into direct subtype relations.

        Plain subtype entries `A` in `KNOWN_SUBTYPES[X]` denote `A ≤ X` and are
        compiled into direct supertype edges. Meet entries are handled
        separately as implication rules because `A & B ≤ X` cannot be reduced to
        direct subtype declarations.
        """
        raw_subtypes = cls.KNOWN_SUBTYPES
        members = frozenset(cls)

        if bad_keys := {node for node in raw_subtypes if node not in members}:
            raise TypeError(f"Expected {cls.__name__} nodes, got {bad_keys!r}.")

        parsed_subtypes: dict[Self, frozenset[Self]] = {}
        for node, subtypes in raw_subtypes.items():
            if bad_subtypes := {
                subtype
                for subtype in subtypes
                if not isinstance(subtype, cls) and not isinstance(subtype, Meet)
            }:
                raise TypeError(
                    f"Expected {cls.__name__} or Meet subtype targets, got"
                    f" {bad_subtypes!r}."
                )
            parsed = frozenset(
                subtype for subtype in subtypes if isinstance(subtype, cls)
            )
            if bad_subtypes := {
                subtype for subtype in parsed if subtype not in members
            }:
                raise TypeError(
                    f"Expected {cls.__name__} subtype targets, got {bad_subtypes!r}."
                )
            parsed_subtypes[node] = parsed

        return parsed_subtypes

    @classmethod
    @cache
    def _validated_subtype_meets(cls) -> tuple[tuple[Self, frozenset[Self]], ...]:
        r"""Validate meet-based subtype implications.

        A declaration `X: {A & B}` in `KNOWN_SUBTYPES` denotes the implication
        `A ∧ B ≤ X`, i.e. every node below all meet factors is also below `X`.
        """
        raw_subtypes = cls.KNOWN_SUBTYPES
        members = frozenset(cls)

        subtype_meets = tuple(
            (node, frozenset(subtype))
            for node, subtypes in raw_subtypes.items()
            for subtype in subtypes
            if isinstance(subtype, Meet)
        )

        all_factors = frozenset().union(*(factors for _, factors in subtype_meets))
        if bad_factors := {factor for factor in all_factors if factor not in members}:
            raise TypeError(
                f"Expected {cls.__name__} subtype-meet factors, got {bad_factors!r}."
            )

        if empty_meets := {node for node, factors in subtype_meets if not factors}:
            raise ValueError(
                f"Expected non-empty subtype-meet factors, got {empty_meets!r}."
            )

        return subtype_meets

    @classmethod
    @cache
    def _validated_supertypes(cls) -> Mapping[Self, frozenset[Self]]:
        r"""Validate compiled supertypes and reject cycles in the order.

        After compilation, every referenced node must still be a member of the
        enum and the resulting direct-supertype graph must be acyclic. The
        returned mapping is the validated adjacency relation used for closure
        computation.
        """
        supertypes = cls._compiled_supertypes()
        members = frozenset(cls)

        if bad_keys := {node for node in supertypes if node not in members}:
            raise TypeError(f"Expected {cls.__name__} nodes, got {bad_keys!r}.")

        all_targets = frozenset().union(*supertypes.values())
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
            for target in supertypes.get(node, frozenset()):
                visit(target)
            stack.remove(node)
            visited.add(node)

        for node in cls:
            visit(node)

        return supertypes

    @classmethod
    @cache
    def _validated_meets(cls) -> tuple[tuple[Self, frozenset[Self]], ...]:
        raw_meets = cls.KNOWN_MEETS
        members = frozenset(cls)

        if bad_keys := {node for node, _ in raw_meets if node not in members}:
            raise TypeError(f"Expected {cls.__name__} meet nodes, got {bad_keys!r}.")

        meets = tuple((node, frozenset(factors)) for node, factors in raw_meets)

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
    def _closure_from(cls, nodes: frozenset[Self], /) -> frozenset[Self]:
        supertypes = cls._validated_supertypes()
        meets = cls._validated_meets()
        subtype_meets = cls._validated_subtype_meets()

        closure: set[Self] = set()
        stack = list(nodes)

        while stack:
            current = stack.pop()
            if current in closure:
                continue

            closure.add(current)
            stack.extend(supertypes.get(current, frozenset()))

            for consequent, factors in meets:
                if consequent not in closure and factors <= closure:
                    stack.append(consequent)

            for consequent, factors in subtype_meets:
                if consequent not in closure and factors <= closure:
                    stack.append(consequent)

        return frozenset(closure)

    def __le__(self, other: object, /) -> bool:
        if isinstance(other, Meet):
            return all(self <= factor for factor in other)
        if not isinstance(other, type(self)):
            return NotImplemented
        return other in self.supertypes

    def __lt__(self, other: object, /) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self <= other and self != other

    @property
    def supertypes(self) -> frozenset[Self]:
        return self._closure_from(frozenset({self}))

    @property
    def factorizations(self) -> frozenset[Meet]:
        return frozenset(
            Meet(factors) for node, factors in self._validated_meets() if node is self
        )

    def __and__(self, other: Self | Meet, /) -> Meet:
        return Meet({self, other})

    def __str__(self) -> str:
        return str(self.value)


PosetEnum.KNOWN_SUPERTYPES = MappingProxyType({})
PosetEnum.KNOWN_SUBTYPES = MappingProxyType({})
PosetEnum.KNOWN_MEETS = ()
