r"""Demonstration of model configuration inference and export."""

from torch import nn

from linodenet.config import infer_blueprint, initialize, validate_blueprint


class TestInitialization:
    def test_linear_model(self) -> None:
        model = nn.Linear(4, 8, bias=False)
        spec = infer_blueprint(model)
        validate_blueprint(model, spec)
        clone = initialize(spec)
        assert isinstance(clone, nn.Linear)
        assert clone.in_features == 4
        assert clone.out_features == 8
        assert clone.bias is None

    def test_sequence_model(self) -> None:
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
        )
        spec = infer_blueprint(model)
        validate_blueprint(model, spec)
        clone = initialize(spec)
        assert isinstance(clone, nn.Sequential)
        assert len(clone) == 2
        assert isinstance(clone[0], nn.Linear)
        assert clone[0].in_features == 4
        assert clone[0].out_features == 8
        assert isinstance(clone[1], nn.ReLU)
