r"""Tests for linodenet.regularizations."""

import pytest

from linodenet.regularizations import (
    FUNCTIONAL_REGULARIZATIONS,
    MODULAR_REGULARIZATIONS,
)
from tests.test_utils import camel2snake, snake2camel


@pytest.mark.parametrize(projection_name=MODULAR_REGULARIZATIONS)
def test_modular(projection_name: str) -> None:
    assert camel2snake(projection_name) in FUNCTIONAL_REGULARIZATIONS


@pytest.mark.parametrize(projection_name=FUNCTIONAL_REGULARIZATIONS)
def test_functional(projection_name: str) -> None:
    assert snake2camel(projection_name) in MODULAR_REGULARIZATIONS
