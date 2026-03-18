r"""Canonical registry for named structural objects."""

__all__ = [
    "Registry",
    "RegistryEntry",
    "REGISTRY",
    "get_registry_entry",
    "normalize_registry_name",
]

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

from linodenet.initializations import INITIALIZATIONS
from linodenet.mappings import PROJECTION_FNS, PROJECTIONS
from linodenet.parametrizations import (
    MATRIX_PARAMETRIZATIONS,
    PARAMETRIZATIONS,
    VECTOR_PARAMETRIZATIONS,
)
from linodenet.regularizations import (
    REGULARIZATION_FNS,
    REGULARIZATION_FNS_WITH_ARGS,
    REGULARIZATIONS,
)
from linodenet.testing import TESTS


@dataclass(slots=True)
class RegistryEntry:
    r"""Connected public objects for a canonical structural name."""

    name: str
    domain: object | None = None
    test: Callable | None = None
    projection: type | None = None
    projection_fn: Callable | None = None
    regularization: type | None = None
    regularization_fn: Callable | None = None
    initialization: Callable | None = None
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
        domain: object | None = None,
        test: object | None = None,
        projection: object | None = None,
        projection_fn: object | None = None,
        regularization: object | None = None,
        regularization_fn: object | None = None,
        initialization: object | None = None,
        parametrization: object | None = None,
    ) -> RegistryEntry:
        r"""Register one or more objects under a canonical name."""
        entry = self._entry_for(name)
        updates = {
            "domain": domain,
            "test": test,
            "projection": projection,
            "projection_fn": projection_fn,
            "regularization": regularization,
            "regularization_fn": regularization_fn,
            "initialization": initialization,
            "parametrization": parametrization,
        }

        for field, value in updates.items():
            if value is None:
                continue
            if getattr(entry, field) is not None:
                canonical_name = normalize_registry_name(name)
                raise ValueError(
                    f"Registry entry {canonical_name!r} already has {field!r} set."
                )
            setattr(entry, field, value)

        return entry

    def register_domain(self, name: str, domain: object, /) -> RegistryEntry:
        r"""Register a domain for `name`."""
        return self.register(name, domain=domain)

    def register_test(self, name: str, test: object, /) -> RegistryEntry:
        r"""Register a test for `name`."""
        return self.register(name, test=test)

    def register_tests(self, tests: Mapping[str, object], /) -> None:
        r"""Register tests from a mapping."""
        for name, test in tests.items():
            self.register_test(name, test)

    def register_projection(self, name: str, projection: type, /) -> RegistryEntry:
        r"""Register a projection module for `name`."""
        return self.register(name, projection=projection)

    def register_projections(self, projections: Mapping[str, type], /) -> None:
        r"""Register projection modules from a mapping."""
        for name, projection in projections.items():
            self.register_projection(name, projection)

    def register_projection_fn(
        self, name: str, projection_fn: object, /
    ) -> RegistryEntry:
        r"""Register a projection function for `name`."""
        return self.register(name, projection_fn=projection_fn)

    def register_projection_fns(self, projection_fns: Mapping[str, object], /) -> None:
        r"""Register projection functions from a mapping."""
        for name, projection_fn in projection_fns.items():
            self.register_projection_fn(name, projection_fn)

    def register_regularization(
        self, name: str, regularization: type, /
    ) -> RegistryEntry:
        r"""Register a regularization module for `name`."""
        return self.register(name, regularization=regularization)

    def register_regularizations(self, regularizations: Mapping[str, type], /) -> None:
        r"""Register regularization modules from a mapping."""
        for name, regularization in regularizations.items():
            self.register_regularization(name, regularization)

    def register_regularization_fn(
        self, name: str, regularization_fn: object, /
    ) -> RegistryEntry:
        r"""Register a regularization function for `name`."""
        return self.register(name, regularization_fn=regularization_fn)

    def register_regularization_fns(
        self, regularization_fns: Mapping[str, object], /
    ) -> None:
        r"""Register regularization functions from a mapping."""
        for name, regularization_fn in regularization_fns.items():
            self.register_regularization_fn(name, regularization_fn)

    def register_initialization(
        self, name: str, initialization: object, /
    ) -> RegistryEntry:
        r"""Register an initialization for `name`."""
        return self.register(name, initialization=initialization)

    def register_initializations(
        self, initializations: Mapping[str, object], /
    ) -> None:
        r"""Register initializations from a mapping."""
        for name, initialization in initializations.items():
            self.register_initialization(name, initialization)

    def register_parametrization(
        self, name: str, parametrization: type, /
    ) -> RegistryEntry:
        r"""Register a parametrization for `name`."""
        return self.register(name, parametrization=parametrization)

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


REGISTRY.register_initializations(INITIALIZATIONS)
REGISTRY.register_tests(TESTS)
REGISTRY.register_parametrizations(PARAMETRIZATIONS)
REGISTRY.register_parametrizations(MATRIX_PARAMETRIZATIONS)
REGISTRY.register_parametrizations(VECTOR_PARAMETRIZATIONS)
REGISTRY.register_regularization_fns(REGULARIZATION_FNS)
REGISTRY.register_regularization_fns(REGULARIZATION_FNS_WITH_ARGS)
REGISTRY.register_regularizations(REGULARIZATIONS)
REGISTRY.register_projections(PROJECTIONS)
REGISTRY.register_projection_fns(PROJECTION_FNS)
