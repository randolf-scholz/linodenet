r"""Tests for the canonical registry."""

from dataclasses import fields
from typing import Any

import pytest

from linodenet.registry import REGISTRY, Registry, RegistryEntry
from tests.testing import pytest_xfail

OPTIONAL_FIELDS = {"name", "initialization"}


@pytest_xfail(condition=lambda name: REGISTRY[name].mapping_fn is None)
@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_registry_entries_are_complete(name: str) -> None:
    r"""Every registry entry should have all public objects populated."""
    entry = REGISTRY[name]
    missing_fields = [
        field.name
        for field in fields(RegistryEntry)
        if field.name not in OPTIONAL_FIELDS and getattr(entry, field.name) is None
    ]
    assert not missing_fields, f"Registry entry {name!r} is missing {missing_fields!r}."


def test_register_rejects_overwriting_existing_field() -> None:
    r"""`Registry.register()` should not overwrite populated fields."""
    registry = Registry()
    old_marker: Any = object()
    registry.register("existing-entry", mapping_fn=old_marker)

    with pytest.raises(ValueError, match=r"already has 'mapping_fn' set"):
        registry.register("existing-entry", mapping_fn=object())  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_update_existing_requires_existing_entry() -> None:
    r"""`Registry.update_existing()` should not create new entries."""
    registry = Registry()

    with pytest.raises(KeyError, match=r"'missing-entry'"):
        registry.update_existing("missing-entry", mapping_fn=object())  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_update_existing_registers_unset_field() -> None:
    r"""`Registry.update_existing()` should fill unset fields on existing entries."""
    registry = Registry()
    marker: Any = object()
    registry.register("existing-entry")

    registry.update_existing("existing-entry", mapping_fn=marker)

    assert registry["existing-entry"].mapping_fn is marker
