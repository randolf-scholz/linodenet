r"""Tests for linodenet.regularizations."""

from linodenet.regularizations import (
    FUNCTIONAL_REGULARIZATIONS,
    MODULAR_REGULARIZATIONS,
)
from tests.test_utils import camel2snake, snake2camel


def test_modular_and_functional():
    for projection_name in MODULAR_REGULARIZATIONS:
        assert camel2snake(projection_name) in FUNCTIONAL_REGULARIZATIONS

    for projection_name in FUNCTIONAL_REGULARIZATIONS:
        assert snake2camel(projection_name) in MODULAR_REGULARIZATIONS
