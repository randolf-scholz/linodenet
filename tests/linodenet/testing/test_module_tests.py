r"""Tests for `linodenet.testing.module_tests`."""

from linodenet.testing.module_tests import assert_forward_stable


def test_forward_stability():
    def identity(x):
        return x

    assert_forward_stable(identity, [(10, 10)], num_runs=100, rtol=1e-3, atol=1e-3)
