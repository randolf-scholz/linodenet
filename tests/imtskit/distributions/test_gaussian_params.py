r"""Gaussian parameter representations, inference, and forwarding methods."""

import pytest
import torch
from torch import Tensor, nn
from torch.utils import _pytree

from imtskit.distributions import MultiHeadGaussian, MultivariateNormal
from imtskit.distributions.gaussian import (
    CholeskyParams,
    CovarianceParams,
    CovarianceType,
    GaussianParams,
    LogCholeskyParams,
    PrecisionParams,
    argmin_forward_kl,
    argmin_reverse_kl,
    fisher,
    inverse_fisher,
    kl,
    log_prob,
)

type _TaggedParams = (
    CovarianceParams | PrecisionParams | CholeskyParams | LogCholeskyParams
)

PARAM_TYPES: dict[CovarianceType, type[_TaggedParams]] = {
    CovarianceType.COVARIANCE: MultivariateNormal.CovarianceParams,
    CovarianceType.PRECISION: MultivariateNormal.PrecisionParams,
    CovarianceType.CHOLESKY: MultivariateNormal.CholeskyParams,
    CovarianceType.LOG_CHOLESKY: MultivariateNormal.LogCholeskyParams,
}
OPERATIONS = [
    "log_prob",
    "kl",
    "fisher",
    "inverse_fisher",
    "argmin_forward_kl",
    "argmin_reverse_kl",
]


@pytest.fixture(params=CovarianceType)
def params(request: pytest.FixtureRequest) -> _TaggedParams:
    r"""Represent the same non-diagonal Gaussian in each coordinate system."""
    covariance_type = CovarianceType(request.param)
    mean = torch.tensor([0.2, -0.3], dtype=torch.float64)
    covariance = torch.tensor([[2.0, 0.4], [0.4, 1.0]], dtype=torch.float64)
    _, matrix = covariance_type.from_covariance((mean, covariance))
    return PARAM_TYPES[covariance_type](mean, matrix)


def _evaluate(
    operation: str,
    theta: GaussianParams,
    other: GaussianParams,
    /,
    *,
    parametrization: str | None,
) -> Tensor | GaussianParams:
    r"""Exercise the shared dispatch contract of the six public operations."""
    value = theta[0] + 0.4
    direction = (
        torch.ones_like(theta[0]),
        torch.eye(theta[0].shape[-1], dtype=theta[0].dtype),
    )
    match operation:
        case "log_prob":
            return log_prob(value, theta, parametrization)
        case "kl":
            return kl(theta, other, parametrization)
        case "fisher":
            return fisher(theta, direction, parametrization)
        case "inverse_fisher":
            return inverse_fisher(theta, direction, parametrization)
        case "argmin_forward_kl":
            return argmin_forward_kl(value, theta, parametrization, retention=0.6)
        case "argmin_reverse_kl":
            return argmin_reverse_kl(value, theta, parametrization, retention=0.6)
        case _:
            raise ValueError(f"Unknown operation {operation!r}.")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("explicit", [False, True])
def test_parametrization_inference(
    params: _TaggedParams, operation: str, *, explicit: bool
) -> None:
    r"""Inferred and matching explicit representations agree with raw-pair calls."""
    other = type(params)(params.mean + 0.3, params[1])
    actual = _evaluate(
        operation,
        params,
        other,
        parametrization=params.covariance_type if explicit else None,
    )
    expected = _evaluate(
        operation,
        (params[0], params[1]),
        (other[0], other[1]),
        parametrization=params.covariance_type,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_plain_pairs_require_parametrization(
    params: _TaggedParams, operation: str
) -> None:
    r"""Plain tensor pairs have no default representation."""
    raw = (params[0], params[1])
    with pytest.raises(ValueError, match="None is not a valid CovarianceType"):
        _evaluate(operation, raw, raw, parametrization=None)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_conflicting_parametrization(params: _TaggedParams, operation: str) -> None:
    r"""Explicit overrides cannot reinterpret tagged matrices."""
    wrong = (
        CovarianceType.PRECISION
        if params.covariance_type == CovarianceType.COVARIANCE
        else CovarianceType.COVARIANCE
    )
    with pytest.raises(ValueError, match="does not match"):
        _evaluate(operation, params, params, parametrization=wrong)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_unknown_parametrization(params: _TaggedParams, operation: str) -> None:
    r"""Tagged inputs still reject invalid representation strings."""
    with pytest.raises(ValueError, match="'unknown' is not a valid CovarianceType"):
        _evaluate(operation, params, params, parametrization="unknown")


def test_kl_checks_both_representations(params: _TaggedParams) -> None:
    r"""KL rejects a conflicting second representation, with or without an override."""
    other = (
        params.to_precision()
        if params.covariance_type == CovarianceType.COVARIANCE
        else params.to_covariance()
    )
    with pytest.raises(ValueError, match="same parametrization"):
        kl(params, other)
    with pytest.raises(ValueError, match="does not match"):
        kl(params, other, params.covariance_type)
    with pytest.raises(ValueError, match="does not match"):
        kl(other, params, params.covariance_type)


def test_kl_plain_pairs_require_explicit_representation(params: _TaggedParams) -> None:
    r"""Each untagged KL operand requires an explicit representation."""
    raw = (params[0], params[1])
    with pytest.raises(ValueError, match="None is not a valid CovarianceType"):
        kl(params, raw)
    with pytest.raises(ValueError, match="None is not a valid CovarianceType"):
        kl(raw, params)
    torch.testing.assert_close(kl(params, raw, params.covariance_type), params.kl(raw))


def test_forwarding_methods(params: _TaggedParams) -> None:
    r"""All instance operations delegate in their own representation."""
    other = type(params)(params.mean + 0.3, params[1])
    value = params.mean + 0.4
    direction = (torch.ones_like(params.mean), torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(params.log_prob(value), log_prob(value, params))
    torch.testing.assert_close(params.kl(other), kl(params, other))
    torch.testing.assert_close(params.fisher(direction), fisher(params, direction))
    torch.testing.assert_close(
        params.inverse_fisher(direction), inverse_fisher(params, direction)
    )
    forward = params.argmin_forward_kl(value, retention=0.6)
    reverse = params.argmin_reverse_kl(value, retention=0.6)
    assert isinstance(forward, type(params))
    assert isinstance(reverse, type(params))
    torch.testing.assert_close(
        tuple(forward), argmin_forward_kl(value, params, retention=0.6)
    )
    torch.testing.assert_close(
        tuple(reverse), argmin_reverse_kl(value, params, retention=0.6)
    )


@pytest.mark.parametrize("target", CovarianceType)
@pytest.mark.parametrize("raw", [False, True])
def test_conversions(
    params: _TaggedParams, target: CovarianceType, *, raw: bool
) -> None:
    r"""Conversions preserve a non-diagonal Gaussian across every representation pair."""
    converted = {
        CovarianceType.COVARIANCE: params.to_covariance,
        CovarianceType.PRECISION: params.to_precision,
        CovarianceType.CHOLESKY: params.to_cholesky,
        CovarianceType.LOG_CHOLESKY: params.to_log_cholesky,
    }[target]()
    assert isinstance(converted, PARAM_TYPES[target])
    assert converted.covariance_type == target
    if target == params.covariance_type:
        assert converted is params
    theta = (params.mean, params[1]) if raw else params
    direct = {
        CovarianceType.COVARIANCE: params.covariance_type.to_covariance,
        CovarianceType.PRECISION: params.covariance_type.to_precision,
        CovarianceType.CHOLESKY: params.covariance_type.to_cholesky,
        CovarianceType.LOG_CHOLESKY: params.covariance_type.to_log_cholesky,
    }[target](theta)
    assert isinstance(direct, PARAM_TYPES[target])
    torch.testing.assert_close(direct, converted)
    expected_covariance = torch.tensor([[2.0, 0.4], [0.4, 1.0]], dtype=torch.float64)
    torch.testing.assert_close(converted.mean, params.mean)
    torch.testing.assert_close(
        converted.to_covariance().covariance, expected_covariance
    )
    reference = torch.distributions.MultivariateNormal(params.mean, expected_covariance)
    value = params.mean + 0.4
    torch.testing.assert_close(converted.log_prob(value), reference.log_prob(value))


@pytest.mark.parametrize("operation", ["argmin_forward_kl", "argmin_reverse_kl"])
def test_mask_forwarding(operation: str) -> None:
    r"""Covariance instance updates forward the observed-coordinate mask."""
    params = MultivariateNormal.CovarianceParams(
        torch.tensor([0.2, -0.3]), torch.tensor([[2.0, 0.4], [0.4, 1.0]])
    )
    value = torch.tensor([0.8, 10.0])
    mask = torch.tensor([True, False])
    actual = getattr(params, operation)(value, retention=0.6, mask=mask)
    function = (
        argmin_forward_kl if operation == "argmin_forward_kl" else argmin_reverse_kl
    )
    expected = function(value, params, retention=0.6, mask=mask)
    assert isinstance(actual, CovarianceParams)
    torch.testing.assert_close(tuple(actual), expected)


@pytest.mark.parametrize("distribution", [MultivariateNormal, MultiHeadGaussian])
def test_parameter_aliases_and_tensor_tree(
    params: _TaggedParams,
    distribution: type[MultivariateNormal | MultiHeadGaussian],
) -> None:
    r"""Both distribution aliases construct pairs with exactly two tensor leaves."""
    parameter_type = getattr(distribution, type(params).__name__)
    pair = parameter_type(*params)
    assert isinstance(pair, type(params))
    assert len(pair) == 2
    leaves, _ = _pytree.tree_flatten(pair)
    assert len(leaves) == 2
    assert leaves[0] is params.mean
    assert leaves[1] is params[1]


class _GaussianLogProb(nn.Module):
    r"""Exercise parameter inference through the instance interface."""

    def forward(self, params: _TaggedParams, value: Tensor) -> Tensor:
        return params.log_prob(value)


class _GaussianConversion(nn.Module):
    r"""Exercise conversions returning parameter tuples through a module boundary."""

    def __init__(self, target: CovarianceType) -> None:
        super().__init__()
        self.method_name = f"to_{target.value.replace('-', '_')}"

    def forward(self, params: _TaggedParams) -> _TaggedParams:
        return getattr(params, self.method_name)()


@pytest.mark.parametrize("target", CovarianceType)
def test_conversion_compile_fullgraph_and_export(
    params: _TaggedParams, target: CovarianceType
) -> None:
    r"""Every conversion exports and compiles while preserving its target type."""
    model = _GaussianConversion(target)
    expected = model(params)
    exported = torch.export.export(model, (params,), strict=True)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    for actual in (exported.module()(params), compiled(params)):
        assert isinstance(actual, PARAM_TYPES[target])
        torch.testing.assert_close(actual, expected)


def test_export(params: _TaggedParams) -> None:
    r"""NamedTuple fields and representation properties work with strict export."""
    model = _GaussianLogProb()
    value = params.mean + 0.4
    exported = torch.export.export(model, (params, value), strict=True)
    actual = exported.module()(params, value)
    torch.testing.assert_close(actual, model(params, value))


def test_compile_fullgraph(params: _TaggedParams) -> None:
    r"""Instance log density compiles with inferred representations and no graph breaks."""
    model = _GaussianLogProb()
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    value = params.mean + 0.4
    torch.testing.assert_close(compiled(params, value), model(params, value))


@pytest.mark.parametrize("operation", ["argmin_forward_kl", "argmin_reverse_kl"])
def test_update_compile_fullgraph(params: _TaggedParams, operation: str) -> None:
    r"""Compiled instance updates retain their representation and tensor-pair structure."""
    update = getattr(params, operation)
    compiled = torch.compile(update, backend="eager", fullgraph=True)
    value = params.mean + 0.4
    expected = update(value, retention=0.6)
    actual = compiled(value, retention=0.6)
    assert isinstance(actual, type(params))
    torch.testing.assert_close(actual, expected)
