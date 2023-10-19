"""Configuration for pytest."""

import argparse

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to make plots.",
    )


@pytest.fixture
def make_plots(request: pytest.FixtureRequest) -> bool:
    """Whether to make plots."""
    return request.config.getoption("--make-plots")
