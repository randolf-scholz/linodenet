r"""Tests for the canonical registry."""

from dataclasses import fields

import pytest

from linodenet.registry import REGISTRY, RegistryEntry

OPTIONAL_FIELDS = {"name", "initialization"}


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
