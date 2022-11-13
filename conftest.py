"""Used for configuring pytest.

https://docs.pytest.org/en/7.1.x/example/parametrize.html#generating-parameters-combinations-depending-on-command-line
"""

import warnings

from pytest import Metafunc, Parser


def pytest_addoption(parser: Parser) -> None:
    parser.addoption("--make_plots", action="store_true", help="Whether to make plots")


def pytest_generate_tests(metafunc: Metafunc) -> None:
    # https://stackoverflow.com/a/42145604/9318372
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    # print(metafunc.config.option)
    # warnings.warn(f"{metafunc.fixturenames=}")
    if "make_plots" in metafunc.fixturenames:
        print(f"{metafunc.fixturenames=}")
        metafunc.parametrize("make_plots", [metafunc.config.option.make_plots])
