r"""Tests for matrix prametrizations."""

import pytest
import torch
from torch import Tensor, nn
from torch._dynamo import OptimizedModule
from torch.fx import GraphModule
from torch.nn.functional import mse_loss
from torch.optim import SGD

from linodenet.mappings.projections import SpectralNorm
from linodenet.parametrize import (
    Banded,
    Diagonal,
    Identity,
    LowerTriangular,
    LowRank,
    Masked,
    ParametrizationBase,
    RankOne,
    SkewSymmetric,
    Symmetric,
    Traceless,
    Tridiagonal,
    UpperTriangular,
    get_parametrizations,
    register_optimizer_hook,
    register_parametrization,
    update_parametrizations,
)
from linodenet.testing import (
    MatrixTest,
    is_banded,
    is_contraction,
    is_diagonal,
    is_low_rank,
    is_lower_triangular,
    is_masked,
    is_rank_one,
    is_skew_symmetric,
    is_symmetric,
    is_traceless,
    is_tridiagonal,
    is_upper_triangular,
)
from tests.testing import DEVICES, TestCase


def is_general_matrix(
    x: Tensor,
    /,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    shape = x.shape[: dim[0]]
    return torch.ones(shape, dtype=torch.bool, device=x.device)


MASK = torch.tensor(
    [
        [True, False, True, False],
        [False, True, False, True],
        [True, True, False, False],
        [False, False, True, True],
    ]
)


MATRIX_PARAMETRIZATIONS: dict[str, nn.Module | type[ParametrizationBase]] = {
    "Banded": Banded(-2, +1),
    "Diagonal": Diagonal,
    "Identity": Identity,
    "LowRank": LowRank(rank=2),
    "LowerTriangular": LowerTriangular,
    "Masked": Masked(mask=MASK),
    "RankOne": RankOne,
    "SkewSymmetric": SkewSymmetric,
    "SpectralNormalization": SpectralNorm(lipschitz_bound=0.97),
    "Symmetric": Symmetric,
    "Traceless": Traceless,
    "Tridiagonal": Tridiagonal,
    "UpperTriangular": UpperTriangular,
}

MATRIX_TESTS: dict[str, MatrixTest] = {
    "Banded": lambda x, **kwargs: is_banded(x, -2, +1, **kwargs),
    "Diagonal": is_diagonal,
    "Identity": is_general_matrix,
    "LowRank": lambda x, **kwargs: is_low_rank(x, rank=2, **kwargs),
    "LowerTriangular": is_lower_triangular,
    "Masked": lambda x, **kwargs: is_masked(x, mask=MASK, **kwargs),
    "RankOne": is_rank_one,
    "SkewSymmetric": is_skew_symmetric,
    "SpectralNormalization": is_contraction,
    "Symmetric": is_symmetric,
    "Traceless": is_traceless,
    "Tridiagonal": is_tridiagonal,
    "UpperTriangular": is_upper_triangular,
}

SHAPES: dict[str, list[tuple[int, int]]] = {
    "Banded": [(5, 4)],
    "Diagonal": [(4, 4)],
    "Identity": [(5, 4)],
    "LowRank": [(5, 4)],
    "LowerTriangular": [(5, 4)],
    "Masked": [(4, 4)],
    "RankOne": [(5, 4)],
    "SkewSymmetric": [(4, 4)],
    "SpectralNormalization": [(5, 4)],
    "Symmetric": [(4, 4)],
    "Traceless": [(4, 4)],
    "Tridiagonal": [(5, 5)],
    "UpperTriangular": [(5, 4)],
}


class TestSuite(TestCase):
    BATCH_SIZE = 8
    VALUE_ATOL = 1e-6
    VALUE_RTOL = 1e-6

    def make_test_case(
        self, shape: tuple[int, int], /, *, device: str
    ) -> tuple[nn.Sequential, Tensor, Tensor]:
        out_features, in_features = shape
        model = nn.Sequential(
            nn.Linear(in_features, in_features, bias=False),
            nn.ReLU(),
            nn.Linear(in_features, out_features, bias=False),
            nn.ReLU(),
            nn.Linear(out_features, out_features, bias=False),
        ).to(device=device)
        x = torch.randn(self.BATCH_SIZE, in_features, device=device)
        y = torch.randn(self.BATCH_SIZE, out_features, device=device)
        return model, x, y

    def get_parametrized_layer(self, model: nn.Module) -> nn.Linear:
        if isinstance(model, OptimizedModule):
            model = model._orig_mod
            assert isinstance(model, nn.Sequential)
            layer = model[2]
            assert isinstance(layer, nn.Linear)
        if isinstance(model, GraphModule):
            children = dict(model.named_children())
            layer = children["2"]
        else:
            assert isinstance(model, nn.Sequential)
            layer = model[2]
            assert isinstance(layer, nn.Linear)
        return layer

    def check_parametrization(self, name: str, model: nn.Module) -> None:
        matrix_test = MATRIX_TESTS[name]
        weight = self.get_parametrized_layer(model).weight
        assert isinstance(weight, Tensor)
        assert matrix_test(weight)

    def assert_stale(self, parametrization: nn.Module, expected: bool) -> None:
        is_stale = getattr(parametrization, "is_stale", None)
        assert is_stale is not None
        assert isinstance(is_stale, Tensor)
        assert is_stale.dtype == torch.bool
        assert bool(is_stale) is expected

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("name", MATRIX_PARAMETRIZATIONS)
    def test_register_parametrization(self, name: str, device: str) -> None:
        shape = SHAPES[name][0]
        model, _, _ = self.make_test_case(shape, device=device)
        layer = self.get_parametrized_layer(model)
        register_parametrization(layer, "weight", MATRIX_PARAMETRIZATIONS[name])

        parametrization = get_parametrizations(layer)["weight"]
        assert isinstance(parametrization, ParametrizationBase)
        assert layer.weight is parametrization.cached_parameter
        self.assert_stale(parametrization, False)
        self.check_parametrization(name, model)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("name", MATRIX_PARAMETRIZATIONS)
    def test_forward_uses_cached_parameter(self, name: str, device: str) -> None:
        torch.manual_seed(0)
        shape = SHAPES[name][0]
        model, x, _ = self.make_test_case(shape, device=device)
        layer = self.get_parametrized_layer(model)
        register_parametrization(layer, "weight", MATRIX_PARAMETRIZATIONS[name])
        parametrization = get_parametrizations(layer)["weight"]
        self.assert_stale(parametrization, False)

        y0 = model(x)
        cached_weight = layer.weight.clone()

        with torch.no_grad():
            parametrization.original_parameter.add_(
                torch.randn_like(parametrization.original_parameter)
            )

        y1 = model(x)
        self.assert_close(y1, y0, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        self.assert_close(
            layer.weight, cached_weight, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL
        )

        update_parametrizations(model)
        self.assert_stale(parametrization, False)
        y2 = model(x)
        assert not torch.allclose(y2, y0)
        self.check_parametrization(name, model)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("name", MATRIX_PARAMETRIZATIONS)
    def test_trainable(self, name: str, device: str) -> None:
        torch.manual_seed(0)
        shape = SHAPES[name][0]
        model, x, y = self.make_test_case(shape, device=device)
        layer = self.get_parametrized_layer(model)
        register_parametrization(layer, "weight", MATRIX_PARAMETRIZATIONS[name])
        optimizer = SGD(model.parameters(), lr=0.1)
        register_optimizer_hook(optimizer, model)
        parametrization = get_parametrizations(layer)["weight"]
        self.assert_stale(parametrization, False)
        original_parameter = parametrization.original_parameter.detach().clone()
        original_output = model(x).detach().clone()

        for _ in range(3):
            model.zero_grad(set_to_none=True)
            loss = mse_loss(model(x), y)
            loss.backward()
            self.assert_stale(parametrization, True)
            optimizer.step()
            self.assert_stale(parametrization, False)
            self.check_parametrization(name, model)

        assert not torch.allclose(
            parametrization.original_parameter, original_parameter
        )
        assert not torch.allclose(model(x), original_output)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("name", MATRIX_PARAMETRIZATIONS)
    def test_compile_forward_uses_cached_parameter(
        self, name: str, device: str
    ) -> None:
        torch.manual_seed(0)
        shape = SHAPES[name][0]
        model, x, _ = self.make_test_case(shape, device=device)
        layer = self.get_parametrized_layer(model)
        register_parametrization(layer, "weight", MATRIX_PARAMETRIZATIONS[name])

        compiled_model = torch.compile(model)
        assert isinstance(compiled_model, OptimizedModule)
        self.check_parametrization(name, compiled_model)

        y0 = compiled_model(x)
        compiled_layer = self.get_parametrized_layer(compiled_model)
        parametrization = get_parametrizations(compiled_layer)["weight"]
        self.assert_stale(parametrization, False)

        with torch.no_grad():
            parametrization.original_parameter.add_(
                torch.randn_like(parametrization.original_parameter)
            )

        y1 = compiled_model(x)
        self.assert_close(y1, y0, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        self.check_parametrization(name, compiled_model)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("name", MATRIX_PARAMETRIZATIONS)
    def test_compile_trainable(self, name: str, device: str) -> None:
        torch.manual_seed(0)
        shape = SHAPES[name][0]
        model, x, y = self.make_test_case(shape, device=device)
        layer = self.get_parametrized_layer(model)
        register_parametrization(layer, "weight", MATRIX_PARAMETRIZATIONS[name])

        compiled_model = torch.compile(model)
        assert isinstance(compiled_model, OptimizedModule)

        optimizer = SGD(compiled_model.parameters(), lr=0.1)
        compiled_layer = self.get_parametrized_layer(compiled_model)
        parametrization = get_parametrizations(compiled_layer)["weight"]
        self.assert_stale(parametrization, False)
        original_parameter = parametrization.original_parameter.detach().clone()

        for _ in range(3):
            compiled_model.zero_grad(set_to_none=True)
            loss = mse_loss(compiled_model(x), y)
            loss.backward()
            self.assert_stale(parametrization, True)
            optimizer.step()
            update_parametrizations(compiled_model)
            self.assert_stale(parametrization, False)
            self.check_parametrization(name, compiled_model)

        assert not torch.allclose(
            parametrization.original_parameter, original_parameter
        )

    @pytest.mark.xfail
    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("name", MATRIX_PARAMETRIZATIONS)
    def test_exported_trainable(self, name: str, device: str) -> None:
        torch.manual_seed(0)
        shape = SHAPES[name][0]
        model, x, y = self.make_test_case(shape, device=device)
        layer = self.get_parametrized_layer(model)
        register_parametrization(layer, "weight", MATRIX_PARAMETRIZATIONS[name])
        parametrization = get_parametrizations(layer)["weight"]
        self.assert_stale(parametrization, False)

        exported_model = torch.export.export(model, args=(x,)).module()
        optimizer = SGD(exported_model.parameters(), lr=0.1)
        exported_layer = self.get_parametrized_layer(exported_model)
        parametrization = get_parametrizations(exported_layer)["weight"]
        self.assert_stale(parametrization, False)
        original_parameter = parametrization.original_parameter.detach().clone()

        for _ in range(3):
            exported_model.zero_grad(set_to_none=True)
            loss = mse_loss(exported_model(x), y)
            loss.backward()
            self.assert_stale(parametrization, True)
            optimizer.step()
            update_parametrizations(exported_model)
            self.assert_stale(parametrization, False)
            self.check_parametrization(name, exported_model)

        assert not torch.allclose(
            parametrization.original_parameter, original_parameter
        )
