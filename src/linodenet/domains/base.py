r"""Base protocols and ordering utilities for domain definitions."""

__all__ = [
    "Domain",
    "DomainMapping",
    "Indeterminate",
    "Join",
    "Meet",
    "Inverse",
    "PosetEnum",
    "ScalarDomain",
    "VectorDomain",
    "MatrixDomain",
    "TensorDomain",
]

from abc import abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache
from types import MappingProxyType, NotImplementedType
from typing import Any, ClassVar, Final, Protocol, Self, overload, runtime_checkable

import torch
from torch import Tensor


@dataclass(frozen=True)
class Indeterminate:
    r"""Placeholder for an order statement whose truth cannot be decided."""

    left: object
    op: str
    right: object

    def __str__(self) -> str:
        return f"{self.left!s} {self.op} {self.right!s}"

    def __bool__(self) -> bool:
        raise TypeError(f"Truth of order statement {self!s} could not be determined.")


def _gt(a, b, /) -> bool | Indeterminate | NotImplementedType:
    try:
        result = a > b
    except TypeError:
        return NotImplemented
    assert (
        result is NotImplemented
        or result is True
        or result is False
        or isinstance(result, Indeterminate)
    )
    return result


def _lt(a, b, /) -> bool | Indeterminate | NotImplementedType:
    try:
        result = a < b
    except TypeError:
        return NotImplemented
    assert (
        result is NotImplemented
        or result is True
        or result is False
        or isinstance(result, Indeterminate)
    )
    return result


def _ge(a, b, /) -> bool | Indeterminate | NotImplementedType:
    try:
        result = a >= b
    except TypeError:
        return NotImplemented
    assert (
        result is NotImplemented
        or result is True
        or result is False
        or isinstance(result, Indeterminate)
    )
    return result


def _le(a, b, /) -> bool | Indeterminate | NotImplementedType:
    try:
        result = a <= b
    except TypeError:
        return NotImplemented
    assert (
        result is NotImplemented
        or result is True
        or result is False
        or isinstance(result, Indeterminate)
    )
    return result


@runtime_checkable
class Domain(Protocol):
    r"""Protocol for Domains."""

    def __contains__(self, value: Tensor, /) -> bool:
        r"""Check if the tensor is in the domain (unbatched only)."""
        return bool(self.check(value).item())

    @abstractmethod
    def check(self, value: Tensor, /) -> Tensor:
        r"""Batched version of contains."""
        raise NotImplementedError

    def __le__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __lt__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __gt__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __ge__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __and__(self, other: Any, /) -> Domain:
        return Meet({self, other})

    def __or__(self, other: Any, /) -> Domain:
        return Join({self, other})


@dataclass(frozen=True)
class Inverse[D: Domain](Domain):
    r"""Structural complement of a domain."""

    domain: Final[D]  # type: ignore[misc]

    def check(self, item: Tensor, /) -> Tensor:
        return ~self.domain.check(item)

    def __contains__(self, item: Tensor, /) -> bool:
        return bool(self.check(item).item())

    def __invert__(self) -> D:
        # (Aᶜ)ᶜ ≡ A
        return self.domain

    def __and__(self, other: Any, /) -> Meet[D]:
        # Aᶜ ∧ B ≡ (Ω-A) ∧ B ≡ (Ω∧B) - (A∧B) ≡ B - (A∧B) ≡ B ∧ (A∧B)ᶜ
        return other & ~(self.domain & other)

    def __or__(self, other: Any, /) -> Join[D]:
        # Aᶜ ∨ B ≡ (Ω-A) ∨ B ≡ (Ω∨B) - (A∨B) ≡ Ω - (A∨B) ≡ (A∨B)ᶜ
        return ~(self.domain | other)


@dataclass(frozen=True)
class Meet[D: Domain](Domain):
    r"""Formal meet expression for the generated lattice.

    When the exact meet is named in the base poset via `KNOWN_MEETS`, order
    comparisons reduce to that node. Otherwise, comparisons use structural meet
    rules only, which keeps results stable when new nodes are added to the
    base poset.
    """

    factors: Final[frozenset[D]]  # type: ignore[misc]

    def __init__(self, args: Iterable[D | Meet[D]] = (), /) -> None:
        nodes: set[D] = set()
        for arg in args:
            match arg:
                case Meet():
                    nodes.update(arg)
                case _:
                    nodes.add(arg)

        object.__setattr__(self, "factors", frozenset(nodes))

    def __contains__(self, item: Tensor, /) -> bool:
        return bool(self.check(item).item())

    def check(self, item: Tensor, /) -> Tensor:
        result = None
        for factor in self:
            factor_result = factor.check(item)
            result = factor_result if result is None else result & factor_result
        return (
            result if result is not None else item.new_full((), True, dtype=torch.bool)
        )

    def __len__(self) -> int:
        return len(self.factors)

    def __iter__(self) -> Iterator[D]:
        return iter(self.factors)

    @overload
    def __and__(self, other: D | Meet[D], /) -> Meet[D]: ...
    @overload
    def __and__(self, other: Join[D], /) -> Join[Domain]: ...
    def __and__(self, other: D | Meet[D] | Join[D], /) -> Meet[D] | Join[Domain]:
        match other:
            case Join():
                # (A₁ ∧ … ∧ Aₙ) ∧ (B₁ ∨ … ∨ Bₙ)
                # ≡ (A₁ ∧ … ∧ Aₙ ∧ B₁) ∨ … ∨ (A₁ ∧ … ∧ Aₙ ∧ Bₙ)
                return Join({self & y for y in other})
            case Meet():
                # (A₁ ∧ … ∧ Aₙ) ∧ (B₁ ∧ … ∧ Bₙ)
                # ≡ (A₁ ∧ … ∧ Aₙ ∧ B₁ ∧ … ∧ Bₙ)
                return Meet({*self, *other})
            case _:
                # (A₁ ∧ … ∧ Aₙ) ∧ B ≡ (A₁ ∧ … ∧ Aₙ ∧ B)
                return Meet({*self, other})

    def __or__(self, other: D | Meet[D] | Join[D], /) -> Meet[Domain]:
        match other:
            case Meet():
                # (A₁ ∧ … ∧ Aₙ) ∨ (B₁ ∧ … ∧ Bₙ)
                # ≡ (A₁ ∨ B) ∧ … ∧ (Aₙ ∨ B)
                # ≡ (A₁ ∨ B₁) ∧ … ∧ (A₁ ∨ Bₙ) ∧ … ∧ (Aₙ ∨ B₁) ∧ … ∧ (Aₙ ∨ Bₙ)
                return Meet({x | y for x in self for y in other})
            case Join():
                # (A₁ ∧ … ∧ Aₙ) ∨ (B₁ ∨ … ∨ Bₙ)
                # ≡ (A₁ ∨ B) ∧ … ∧ (Aₙ ∨ B)
                # ≡ (A₁ ∨ B₁ ∨ … ∨ Bₙ) ∧ … ∧ (Aₙ ∨ B₁ ∨ … ∨ Bₙ)
                return Meet({m | other for m in self})
            case _:
                # (A₁ ∧ … ∧ Aₙ) ∨ B ≡ (A₁ ∨ B) ∧ … ∧ (Aₙ ∨ B)
                return Meet({m | other for m in self})

    def __le__(self, other: object, /) -> bool | Indeterminate:
        # (A₁ ∧ … ∧ Aₙ) ≤ B ⇐ Aᵢ ≤ B for some i (sufficient condition)
        if any(_le(member, other) is True for member in self):
            return True
        return NotImplemented

    def __lt__(self, other: object, /) -> bool | Indeterminate:
        # (A₁ ∧ … ∧ Aₙ) < B ⇐ Aᵢ < B for some i (sufficient condition)
        if any(_lt(member, other) is True for member in self):
            return True
        return NotImplemented

    def __ge__(self, other: object, /) -> bool | Indeterminate:
        # B ≤ (A₁ ∧ … ∧ Aₙ) ⟺ B ≤ A₁ and … and B ≤ Aₙ
        for factor in self:
            match _le(other, factor):
                case False:
                    return False
                case True:
                    continue
                case _:
                    return Indeterminate(other, "<=", self)
        return True

    def __gt__(self, other: object, /) -> bool | Indeterminate:
        return self >= other and self != other


@dataclass(frozen=True)
class Join[D: Domain](Domain):
    r"""Formal join expression for the generated lattice.

    Order comparisons use structural join rules only. This keeps results stable
    when new nodes are added to the base poset.
    """

    members: Final[frozenset[D]]  # type: ignore[misc]

    def __init__(self, args: Iterable[D | Join[D]] = (), /) -> None:
        nodes: set[D] = set()
        for arg in args:
            match arg:
                case Join():
                    nodes.update(arg)
                case _:
                    nodes.add(arg)

        object.__setattr__(self, "members", frozenset(nodes))

    def __contains__(self, item: Tensor, /) -> bool:
        return bool(self.check(item).item())

    def check(self, item: Tensor, /) -> Tensor:
        result = None
        for member in self:
            member_result = member.check(item)
            result = member_result if result is None else result | member_result
        return (
            result if result is not None else item.new_full((), False, dtype=torch.bool)
        )

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self) -> Iterator[D]:
        return iter(self.members)

    def __and__(self, other: D | Join[D] | Meet[D], /) -> Join[Domain]:
        match other:
            case Join():
                # (A₁ ∨ … ∨ Aₙ) ∧ (B₁ ∨ … ∨ Bₙ)
                # ≡ (A₁ ∧ B₁) ∨ … ∨ (A₁ ∧ Bₙ) ∨ … ∨ (Aₙ ∧ B₁) ∨ … ∨ (Aₙ ∧ Bₙ)
                return Join({x & y for x in self for y in other})
            case Meet():
                # (A₁ ∨ … ∨ Aₙ) ∧ (B₁ ∧ … ∧ Bₙ)
                # ≡ (A₁ ∧ B₁ ∧ … ∧ Bₙ) ∨ … ∨ (Aₙ ∧ V₁ ∧ … ∧ Bₙ)
                return Join({x & other for x in self})
            case _:
                # (A₁ ∨ … ∨ Aₙ) ∧ B
                # ≡ (A₁ ∧ B) ∨ … ∨ (Aₙ ∧ B)
                return Join({x & other for x in self})

    @overload
    def __or__(self, other: D | Join[D], /) -> Join[D]: ...
    @overload
    def __or__(self, other: Meet[D], /) -> Meet[Domain]: ...
    def __or__(self, other: D | Join[D] | Meet[D], /) -> Join[D] | Meet[Domain]:
        match other:
            case Join():
                # (A₁ ∨ … ∨ Aₙ) ∨ (B₁ ∨ … ∨ Bₙ) ≡ (A₁ ∨ … ∨ Aₙ ∨ B₁ ∨ … ∨ Bₙ)
                return Join({*self, *other})
            case Meet():
                # (A₁ ∨ … ∨ Aₙ) ∨ (B₁ ∧ … ∧ Bₙ)
                # (A₁ ∨ … ∨ Aₙ ∨ B₁) ∧ … ∧ (A₁ ∨ … ∨ Aₙ ∨ Bₙ)
                return Meet({self | y for y in other})
            case _:
                # (A₁ ∨ … ∨ Aₙ) ∨ B ≡ (A₁ ∨ … ∨ Aₙ ∨ B)
                return Join({*self, other})

    def __le__(self, other: object, /) -> bool | Indeterminate:
        # (A₁ ∨ … ∨ Aₙ) ≤ B ⟺ A₁ ≤ B ∧ … ∧ Aₙ ≤ B
        for member in self:
            match _le(member, other):
                case False:
                    return False
                case True:
                    continue
                case _:
                    return Indeterminate(self, "<=", other)
        return True

    def __lt__(self, other: object, /) -> bool | Indeterminate:
        return self <= other and self != other

    def __ge__(self, other: object, /) -> bool | Indeterminate:
        # B ≤ (A₁ ∨ … ∨ Aₙ) ⇐ B ≤ Aᵢ for some i (sufficient condition)
        if any(_ge(member, other) is True for member in self):
            return True
        if isinstance(other, Domain):
            return Indeterminate(other, "<=", self)
        return NotImplemented

    def __gt__(self, other: object, /) -> bool | Indeterminate:
        # B < (A₁ ∨ … ∨ Aₙ) ⇐ B < Aᵢ for some i (sufficient condition)
        if any(_gt(member, other) is True for member in self):
            return True
        return NotImplemented


@dataclass(frozen=True)
class DomainMapping[D: Domain](Mapping[D, D]):
    r"""Immutable monotone mapping between domains.

    Lookups first try the exact key. If the key is absent, the mapping falls
    back to the unique least stored upper bound of the requested key.
    """

    domains: Final[Mapping[D, D]]  # type: ignore[misc]

    def __init__(self, domains: Mapping[D, D], /) -> None:
        backend = dict(domains)
        self._validate(backend)
        object.__setattr__(self, "domains", MappingProxyType(backend))

    def __len__(self) -> int:
        return len(self.domains)

    def __iter__(self) -> Iterator[D]:
        return iter(self.domains)

    @staticmethod
    def _validate(domains: Mapping[D, D], /) -> None:
        items = tuple(domains.items())
        for domain, codomain in items:
            if type(domain) is not type(codomain):
                raise TypeError(
                    "Expected domain and codomain to have the same type, got "
                    f"{type(domain)!r} and {type(codomain)!r}."
                )

        for left_domain, left_codomain in items:
            for right_domain, right_codomain in items:
                if left_domain <= right_domain and not left_codomain <= right_codomain:
                    raise ValueError(
                        "Expected a monotone domain mapping, but "
                        f"{left_domain!r} <= {right_domain!r} while "
                        f"{left_codomain!r} ≰ {right_codomain!r}."
                    )

    def __getitem__(self, key: D, /) -> D | Join:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        if key in self.domains:
            return self.domains[key]

        upper_bounds = {candidate for candidate in self.domains if key <= candidate}
        if not upper_bounds:
            raise KeyError(key)

        least_upper_bounds = {
            candidate
            for candidate in upper_bounds
            if all(
                other == candidate or not other <= candidate for other in upper_bounds
            )
        }
        if not least_upper_bounds:
            raise KeyError(key)
        if len(least_upper_bounds) != 1:
            return Join(self.domains[candidate] for candidate in least_upper_bounds)

        lub = next(iter(least_upper_bounds))
        return self.domains[lub]


class PosetEnum(Enum):
    r"""Mixin implementing a partial order from immediate-superset edges."""

    KNOWN_SUPERTYPES: ClassVar[Mapping[Self, frozenset[Self | Meet[Self]]]]
    r"""Dependencies"""
    KNOWN_SUBTYPES: ClassVar[Mapping[Self, frozenset[Self | Meet[Self]]]]
    r"""Reverse dependencies."""
    KNOWN_MEETS: ClassVar[Sequence[tuple[Self, Meet[Self]]]]
    r"""Named meet rules encoded as implications x≤aᵢ ∀i ⇒ x≤m."""

    def __contains__(self, value: Tensor, /) -> bool:
        return bool(self.check(value).item())

    @abstractmethod
    def check(self, value: Tensor, /) -> Tensor:
        raise NotImplementedError

    def __le__(self, other: object, /) -> bool | Indeterminate:
        if isinstance(other, type(self)):
            return other in self.supertypes
        if isinstance(other, Meet):
            for factor in other:
                match _le(self, factor):
                    case False:
                        return False
                    case True:
                        continue
                    case _:
                        return Indeterminate(self, "<=", other)
            return True
        if isinstance(other, Join):
            if any(_le(self, member) is True for member in other):
                return True
            return Indeterminate(self, "<=", other)
        return NotImplemented

    def __lt__(self, other: object, /) -> bool | Indeterminate:
        if isinstance(other, type(self)):
            return self <= other and self != other
        return NotImplemented

    def __ge__(self, other: object, /) -> bool | Indeterminate:
        if isinstance(other, type(self)):
            return self in other.supertypes
        if isinstance(other, Meet):
            if all(isinstance(factor, type(self)) for factor in other):
                return self in type(self)._closure_from(frozenset(other))
            if any(_ge(self, factor) is True for factor in other):
                return True
            return Indeterminate(self, ">=", other)
        if isinstance(other, Join):
            for member in other:
                match _le(member, self):
                    case False:
                        return False
                    case True:
                        continue
                    case _:
                        return Indeterminate(self, ">=", other)
            return True
        return NotImplemented

    def __gt__(self, other: object, /) -> bool | Indeterminate:
        if isinstance(other, type(self)):
            return self >= other and self != other
        return NotImplemented

    def __and__(self, other: Self | Meet[Self], /) -> Meet[Self]:
        return Meet({self, other})

    def __or__(self, other: Self | Join[Self], /) -> Join[Self]:
        return Join({self, other})

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
    def _parse_known_supertypes(cls) -> Mapping[Self, frozenset[Self]]:
        r"""Parse declared supertypes into direct supertype relations.

        `KNOWN_SUPERTYPES` may contain plain nodes or meet expressions. A meet
        target `A & B` denotes the stronger statement `x ≤ A ∧ B`, so this
        parser expands it to the implied direct supertypes `A` and `B`.
        """
        raw_supers: Mapping[Self, frozenset[Self | Meet[Self]]] = cls.KNOWN_SUPERTYPES  # type: ignore[assignment]
        members = frozenset(cls)

        if bad_keys := {node for node in raw_supers if node not in members}:
            raise TypeError(f"Expected {cls.__name__} nodes, got {bad_keys!r}.")

        supertypes: dict[Self, frozenset[Self]] = {}
        for node, supers in raw_supers.items():
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
    def _parse_known_subtypes(
        cls,
    ) -> tuple[
        Mapping[Self, frozenset[Self]], tuple[tuple[Self, frozenset[Self]], ...]
    ]:
        r"""Parse declared subtype relations from `KNOWN_SUBTYPES`.

        Plain subtype entries `A` in `KNOWN_SUBTYPES[X]` denote `A ≤ X` and are
        compiled into direct supertype edges. Meet entries are handled
        separately as implication rules because `A & B ≤ X` cannot be reduced to
        direct subtype declarations.
        """
        raw_subtypes: Mapping[Self, frozenset[Self | Meet[Self]]] = cls.KNOWN_SUBTYPES  # type: ignore[assignment]
        members = frozenset(cls)

        if bad_keys := {node for node in raw_subtypes if node not in members}:
            raise TypeError(f"Expected {cls.__name__} nodes, got {bad_keys!r}.")

        parsed_subtypes: dict[Self, frozenset[Self]] = {}
        subtype_meets: list[tuple[Self, frozenset[Self]]] = []
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
            subtype_meets.extend(
                (node, factors)
                for subtype in subtypes
                if isinstance(subtype, Meet)
                for factors in (frozenset(subtype),)
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

        return parsed_subtypes, tuple(subtype_meets)

    @classmethod
    @cache
    def _parse_known_subtype_edges(cls) -> Mapping[Self, frozenset[Self]]:
        r"""Parse declared plain subtypes into direct subtype relations."""
        parsed_subtypes, _ = cls._parse_known_subtypes()
        return parsed_subtypes

    @classmethod
    @cache
    def _parse_known_subtype_meets(cls) -> tuple[tuple[Self, frozenset[Self]], ...]:
        r"""Parse declared meet-based subtype implications.

        A declaration `X: {A & B}` in `KNOWN_SUBTYPES` denotes the implication
        `A ∧ B ≤ X`, i.e. every node below all meet factors is also below `X`.
        """
        _, subtype_meets = cls._parse_known_subtypes()
        return subtype_meets

    @classmethod
    @cache
    def _parse_known_meets(cls) -> tuple[tuple[Self, frozenset[Self]], ...]:
        raw_meets: Sequence[tuple[Self, Meet[Self]]] = cls.KNOWN_MEETS  # type: ignore[assignment]
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
    def _compiled_supertypes(cls) -> Mapping[Self, frozenset[Self]]:
        r"""Compile all declared supertype information into one direct map.

        This combines parsed `KNOWN_SUPERTYPES`, reverse declarations from
        `KNOWN_SUBTYPES`, structural supertype relations induced by
        `KNOWN_MEETS`, and the implicit top/bottom node conventions.

        The result still contains only direct supertype edges. Transitive
        closure is computed separately by `_closure_from`.
        """
        supertypes: dict[Self, set[Self]] = {node: set() for node in cls}

        for node, supers in cls._parse_known_supertypes().items():
            supertypes[node].update(supers)

        for meet, factors in cls._parse_known_meets():
            supertypes[meet].update(factors)

        for node, subtypes in cls._parse_known_subtype_edges().items():
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
    def _closure_from(cls, nodes: frozenset[Self], /) -> frozenset[Self]:
        supertypes = cls._validated_supertypes()
        meets = cls._parse_known_meets()
        subtype_meets = cls._parse_known_subtype_meets()

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

    @property
    def supertypes(self) -> frozenset[Self]:
        return self._closure_from(frozenset({self}))

    @property
    def factorizations(self) -> frozenset[Meet[Self]]:
        return frozenset(
            Meet(factors) for node, factors in self._parse_known_meets() if node is self
        )

    def __str__(self) -> str:
        return str(self.value)


PosetEnum.KNOWN_SUPERTYPES = MappingProxyType({})  # pyright: ignore[reportAttributeAccessIssue]
PosetEnum.KNOWN_SUBTYPES = MappingProxyType({})  # pyright: ignore[reportAttributeAccessIssue]
PosetEnum.KNOWN_MEETS = ()


class ScalarDomain:
    r"""Base class for scalar domains."""

    @property
    def shape(self) -> tuple[()]:
        return ()

    @abstractmethod
    def check(self, value: Tensor, /) -> Tensor:
        raise NotImplementedError

    def __contains__(self, value: Tensor, /) -> bool:
        return bool(self.check(value).item())

    def __le__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __lt__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __gt__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __ge__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __or__(self, other: ScalarDomain, /) -> ScalarDomain | Join[ScalarDomain]:
        return Join({self, other})

    def __and__(self, other: ScalarDomain, /) -> ScalarDomain | Meet[ScalarDomain]:
        return Meet({self, other})


class VectorDomain:
    r"""Base class for vector domains."""

    @property
    @abstractmethod
    def size(self) -> int | None: ...

    @property
    def shape(self) -> tuple[int] | None:
        return None if self.size is None else (self.size,)

    @abstractmethod
    def check(self, value: Tensor, /) -> Tensor:
        raise NotImplementedError

    def __contains__(self, value: Tensor, /) -> bool:
        return bool(self.check(value).item())

    def __le__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __lt__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __gt__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __ge__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __or__(self, other: VectorDomain, /) -> VectorDomain | Join[VectorDomain]:
        return Join({self, other})

    def __and__(self, other: VectorDomain, /) -> VectorDomain | Meet[VectorDomain]:
        return Meet({self, other})


class MatrixDomain:
    r"""Stub base class for matrix domains."""

    @property
    @abstractmethod
    def rows(self) -> int | None: ...

    @property
    @abstractmethod
    def cols(self) -> int | None: ...

    @property
    def shape(self) -> tuple[int, int] | None:
        if self.rows is not None and self.cols is not None:
            return self.rows, self.cols
        return None

    @abstractmethod
    def check(self, value: Tensor, /) -> Tensor:
        raise NotImplementedError

    def __contains__(self, value: Tensor, /) -> bool:
        return bool(self.check(value).item())

    def __le__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __lt__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __gt__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __ge__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __or__(self, other: MatrixDomain, /) -> MatrixDomain | Join[MatrixDomain]:
        return Join({self, other})

    def __and__(self, other: MatrixDomain, /) -> MatrixDomain | Meet[MatrixDomain]:
        return Meet({self, other})


class TensorDomain:
    r"""Base class for tensor domains."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...] | None: ...

    @abstractmethod
    def check(self, value: Tensor, /) -> Tensor:
        raise NotImplementedError

    def __contains__(self, value: Tensor, /) -> bool:
        return bool(self.check(value).item())

    def __le__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __lt__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __gt__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __ge__(self, other: Any, /) -> bool | Indeterminate:
        return NotImplemented

    def __or__(self, other: TensorDomain, /) -> TensorDomain | Join[TensorDomain]:
        return Join({self, other})

    def __and__(self, other: TensorDomain, /) -> TensorDomain | Meet[TensorDomain]:
        return Meet({self, other})
