r"""Configuration for pytest."""

import argparse

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    r"""Add options to pytest."""
    parser.addoption(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to make plots.",
    )


@pytest.fixture
def make_plots(request: pytest.FixtureRequest) -> bool:
    r"""Whether to make plots."""
    return bool(request.config.getoption("--make-plots"))
