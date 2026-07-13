r"""Canonical registry for named structural objects."""

__all__ = [
    "Registry",
    "RegistryEntry",
    "REGISTRY",
    "get_registry_entry",
]

from collections.abc import Callable as Fn, Iterator, Mapping
from dataclasses import dataclass, fields

from .domains import MATRIX_TESTS, VECTOR_TESTS, Domain, is_orthogonal
from .initializations import INITIALIZATION_FNS
from .mappings import BIJECTIONS, PROJECTION_FNS, SURJECTIONS
from .parametrizations import (
    MATRIX_PARAMETRIZATIONS,
    VECTOR_PARAMETRIZATIONS,
)
from .regularizations import (
    REGULARIZATION_FNS_WITH_ARGS,
    REGULARIZATION_FNS_WITHOUT_ARGS,
    REGULARIZATIONS,
)
from .utils import normalize_name


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


class Registry(Mapping[str, RegistryEntry]):
    r"""Mutable registry keyed by canonical lowercase kebab-case names."""

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def _entry_for(self, name: str, /) -> RegistryEntry:
        canonical_name = normalize_name(name)
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
        canonical_name = normalize_name(name)
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

    def update_existing(
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
        r"""Register objects on an existing canonical name."""
        canonical_name = normalize_name(name)
        if canonical_name not in self._entries:
            raise KeyError(f"Registry entry {canonical_name!r} does not exist.")

        self.register(
            canonical_name,
            domain=domain,
            test=test,
            mapping=mapping,
            mapping_fn=mapping_fn,
            regularization=regularization,
            regularization_fn=regularization_fn,
            initialization=initialization,
            parametrization=parametrization,
        )

    def register_test_on_domain(self, domain: Domain, test: Fn, /) -> None:
        r"""Register a test for `domain`."""
        for entry in self._entries.values():
            if entry.domain != domain:
                continue
            if entry.test is not None and entry.test is not test:
                raise ValueError(
                    f"Registry entry {entry.name!r} already has 'test' set."
                )
            self.register(entry.name, test=test)

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
        for domain, test in tests.items():
            self.register_test_on_domain(domain, test)

    def register_mappings(self, mappings: Mapping[str, type], /) -> None:
        for name, projection in mappings.items():
            self.register(name, mapping=projection)

    def register_mapping_fns(self, fns: Mapping[str, Fn], /) -> None:
        for name, fn in fns.items():
            self.register(name, mapping_fn=fn)

    def register_regularizations(self, typs: Mapping[str, type], /) -> None:
        for name, regularization in typs.items():
            self.register(name, regularization=regularization)

    def register_regularization_fns(self, fns: Mapping[str, Fn], /) -> None:
        for name, fn in fns.items():
            self.register(name, regularization_fn=fn)

    def register_initializations(self, fns: Mapping[str, Fn], /) -> None:
        for name, initialization in fns.items():
            self.register(name, initialization=initialization)

    def register_parametrizations(self, typs: Mapping[str, type], /) -> None:
        for name, parametrization in typs.items():
            self.register(name, parametrization=parametrization)

    def __getitem__(self, key: str) -> RegistryEntry:
        return self._entries[normalize_name(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def get[T = None](self, key: str, default: T = None) -> RegistryEntry | T:  # type: ignore[assignment]
        r"""Return the entry for `key` if present."""
        return self._entries.get(normalize_name(key), default)


REGISTRY = Registry()
r"""Canonical registry of named structural objects."""


def get_registry_entry(name: str, /) -> RegistryEntry:
    r"""Return the registry entry associated with `name`."""
    return REGISTRY[name]


REGISTRY.register_mappings(SURJECTIONS)
REGISTRY.register_mappings(BIJECTIONS)
REGISTRY.register_mapping_fns(PROJECTION_FNS)
REGISTRY.register_initializations(INITIALIZATION_FNS)
REGISTRY.register_parametrizations(MATRIX_PARAMETRIZATIONS)
REGISTRY.register_parametrizations(VECTOR_PARAMETRIZATIONS)
REGISTRY.register_regularization_fns(REGULARIZATION_FNS_WITHOUT_ARGS)
REGISTRY.register_regularization_fns(REGULARIZATION_FNS_WITH_ARGS)
REGISTRY.register_regularizations(REGULARIZATIONS)
REGISTRY.register_tests_on_domain(MATRIX_TESTS)  # must be registered last
REGISTRY.register_tests_on_domain(VECTOR_TESTS)  # must be registered last

# overrides
REGISTRY.update_existing("orthogonal-cayley", test=is_orthogonal)
REGISTRY.update_existing("cayley-map", test=is_orthogonal)
