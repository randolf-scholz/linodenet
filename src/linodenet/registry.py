r"""Canonical registry for named structural objects."""

__all__ = [
    "Registry",
    "RegistryEntry",
    "REGISTRY",
    "get_registry_entry",
    "normalize_registry_name",
]

from collections.abc import Callable as Fn, Iterator, Mapping
from dataclasses import dataclass, fields

from linodenet.domains import Domain
from linodenet.initializations import INITIALIZATIONS
from linodenet.mappings import BIJECTIONS, PROJECTION_FNS, SURJECTIONS
from linodenet.parametrizations import (
    MATRIX_PARAMETRIZATIONS,
    PARAMETRIZATIONS,
    VECTOR_PARAMETRIZATIONS,
)
from linodenet.regularizations import (
    REGULARIZATION_FNS_WITH_ARGS,
    REGULARIZATION_FNS_WITHOUT_ARGS,
    REGULARIZATIONS,
)
from linodenet.testing import MATRIX_TESTS, VECTOR_TESTS


@dataclass(slots=True)
class RegistryEntry:
    r"""Connected public objects for a canonical structural name."""

    name: str
    domain: Domain | None = None
    test: Fn | None = None
    mapping: type | None = None
    mapping_fn: Fn | None = None
    regularization: type | None = None
    regularization_fn: Fn | None = None
    initialization: Fn | None = None
    parametrization: type | None = None


def _camel_to_snake(name: str, /) -> str:
    r"""Convert `CamelCase` names to `snake_case`."""
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


def _snake_to_kebab(string: str) -> str:
    return string.replace("_", "-").lower()


def normalize_registry_name(name: str, /) -> str:
    r"""Normalize names to lowercase kebab-case."""
    name = name.removeprefix("is_")
    name = _camel_to_snake(name)
    name = _snake_to_kebab(name)
    return name


class Registry(Mapping[str, RegistryEntry]):
    r"""Mutable registry keyed by canonical lowercase kebab-case names."""

    _entries: dict[str, RegistryEntry]

    def __init__(self) -> None:
        self._entries = {}

    def _entry_for(self, name: str, /) -> RegistryEntry:
        canonical_name = normalize_registry_name(name)
        if canonical_name not in self._entries:
            self._entries[canonical_name] = RegistryEntry(name=canonical_name)
        return self._entries[canonical_name]

    def register(
        self,
        name: str,
        /,
        *,
        domain: Domain | None = None,
        test: Fn | None = None,
        mapping: type | None = None,
        mapping_fn: Fn | None = None,
        regularization: type | None = None,
        regularization_fn: Fn | None = None,
        initialization: Fn | None = None,
        parametrization: type | None = None,
    ) -> None:
        r"""Register one or more objects under a canonical name."""
        entry = self._entry_for(name)
        canonical_name = normalize_registry_name(name)
        candidate = RegistryEntry(
            name=canonical_name,
            domain=domain,
            test=test,
            mapping=mapping,
            mapping_fn=mapping_fn,
            regularization=regularization,
            regularization_fn=regularization_fn,
            initialization=initialization,
            parametrization=parametrization,
        )

        inferred_domain = (
            getattr(mapping, "CODOMAIN", None) if mapping is not None else None
        )
        if inferred_domain is not None:
            current_domain = (
                candidate.domain if candidate.domain is not None else entry.domain
            )
            if current_domain is None:
                candidate.domain = inferred_domain
            elif current_domain != inferred_domain:
                raise ValueError(
                    f"Registry entry {canonical_name!r} has inconsistent 'domain': "
                    f"{current_domain!r} != {inferred_domain!r}."
                )

        for field in fields(RegistryEntry):
            if field.name == "name":
                continue
            if (value := getattr(candidate, field.name)) is None:
                continue
            if getattr(entry, field.name) is not None:
                raise ValueError(
                    f"Registry entry {canonical_name!r} already has {field.name!r} set."
                )
            setattr(entry, field.name, value)

    def register_domain(self, name: str, domain: Domain, /) -> None:
        r"""Register a domain for `name`."""
        self.register(name, domain=domain)

    def register_test_on_domain(self, domain: Domain, test: Fn, /) -> None:
        r"""Register a test for `domain`."""
        for entry in self._entries.values():
            if entry.domain != domain:
                continue
            if entry.test is not None and entry.test is not test:
                raise ValueError(
                    f"Registry entry {entry.name!r} already has 'test' set."
                )
            entry.test = test

        test_name = test.__name__
        assert test_name.startswith("is_")
        name = test_name.removeprefix("is_")
        if name not in self:
            self.register(name, domain=domain, test=test)
        else:
            existing = self[name]
            if existing.domain is None:
                self.register(name, domain=domain)
            if existing.test is None:
                self.register(name, test=test)

    def register_tests_on_domain[D: Domain](self, tests: Mapping[D, Fn], /) -> None:
        r"""Register tests from a domain-to-test mapping."""
        for domain, test in tests.items():
            self.register_test_on_domain(domain, test)

    def register_projection(self, name: str, projection: type, /) -> None:
        r"""Register a projection module for `name`."""
        self.register(name, mapping=projection)

    def register_mappings(self, projections: Mapping[str, type], /) -> None:
        r"""Register projection modules from a mapping."""
        for name, projection in projections.items():
            self.register_projection(name, projection)

    def register_projection_fn(self, name: str, fn: Fn, /) -> None:
        r"""Register a projection function for `name`."""
        self.register(name, mapping_fn=fn)

    def register_mapping_fns(self, fns: Mapping[str, Fn], /) -> None:
        r"""Register projection functions from a mapping."""
        for name, projection_fn in fns.items():
            self.register_projection_fn(name, projection_fn)

    def register_regularization(self, name: str, typ: type, /) -> None:
        r"""Register a regularization module for `name`."""
        self.register(name, regularization=typ)

    def register_regularizations(self, typs: Mapping[str, type], /) -> None:
        r"""Register regularization modules from a mapping."""
        for name, regularization in typs.items():
            self.register_regularization(name, regularization)

    def register_regularization_fn(self, name: str, fn: Fn, /) -> None:
        r"""Register a regularization function for `name`."""
        self.register(name, regularization_fn=fn)

    def register_regularization_fns(self, fns: Mapping[str, Fn], /) -> None:
        r"""Register regularization functions from a mapping."""
        for name, regularization_fn in fns.items():
            self.register_regularization_fn(name, regularization_fn)

    def register_initialization(self, name: str, fn: Fn, /) -> None:
        r"""Register an initialization for `name`."""
        self.register(name, initialization=fn)

    def register_initializations(self, fns: Mapping[str, Fn], /) -> None:
        r"""Register initializations from a mapping."""
        for name, initialization in fns.items():
            self.register_initialization(name, initialization)

    def register_parametrization(self, name: str, parametrization: type, /) -> None:
        r"""Register a parametrization for `name`."""
        self.register(name, parametrization=parametrization)

    def register_parametrizations(
        self, parametrizations: Mapping[str, type], /
    ) -> None:
        r"""Register parametrizations from a mapping."""
        for name, parametrization in parametrizations.items():
            self.register_parametrization(name, parametrization)

    def __getitem__(self, key: str) -> RegistryEntry:
        return self._entries[normalize_registry_name(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def get[T = None](self, key: str, default: T = None) -> RegistryEntry | T:  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]
        r"""Return the entry for `key` if present."""
        return self._entries.get(normalize_registry_name(key), default)


REGISTRY = Registry()
r"""Canonical registry of named structural objects."""


def get_registry_entry(name: str, /) -> RegistryEntry:
    r"""Return the registry entry associated with `name`."""
    return REGISTRY[name]


REGISTRY.register_mappings(SURJECTIONS)
REGISTRY.register_mappings(BIJECTIONS)
REGISTRY.register_mapping_fns(PROJECTION_FNS)
REGISTRY.register_initializations(INITIALIZATIONS)
REGISTRY.register_parametrizations(PARAMETRIZATIONS)
REGISTRY.register_parametrizations(MATRIX_PARAMETRIZATIONS)
REGISTRY.register_parametrizations(VECTOR_PARAMETRIZATIONS)
REGISTRY.register_regularization_fns(REGULARIZATION_FNS_WITHOUT_ARGS)
REGISTRY.register_regularization_fns(REGULARIZATION_FNS_WITH_ARGS)
REGISTRY.register_regularizations(REGULARIZATIONS)
REGISTRY.register_tests_on_domain(MATRIX_TESTS)  # must be registered last
REGISTRY.register_tests_on_domain(VECTOR_TESTS)  # must be registered last
