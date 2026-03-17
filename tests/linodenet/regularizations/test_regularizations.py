r"""Tests for linodenet.regularizations."""

import pytest

from linodenet.regularizations import REGULARIZATIONS
from tests.testing import camel2snake, snake2camel


@pytest.mark.parametrize("projection_name", REGULARIZATIONS)
def test_modular(projection_name: str) -> None:
    assert camel2snake(projection_name) in REGULARIZATIONS
    assert snake2camel(projection_name) in REGULARIZATIONS
