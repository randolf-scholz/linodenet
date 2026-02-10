r"""Demonstration of model configuration inference and export."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import Tensor, nn

from linodenet.config import infer_blueprint, validate_blueprint
from linodenet.serialization import deserialize_model, serialize_model


class TestSerialization:
    def test_linear(self) -> None:
        model = nn.Linear(4, 8, bias=False)
        spec = infer_blueprint(model)
        validate_blueprint(model, spec)
        serialize_model(model, "model.zip")
        deserialized = deserialize_model("model.zip")
        assert isinstance(deserialized, nn.Linear)
        assert deserialized.in_features == 4
        assert deserialized.out_features == 8
        assert deserialized.bias is None

    def test_sequential(self) -> None:
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
        )
        spec = infer_blueprint(model)
        validate_blueprint(model, spec)

        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            spec_path = Path(tmpdir) / "config.json"
            serialized_spec = serialize_model(model, model_path)

            with spec_path.open("w", encoding="utf-8") as file:
                json.dump(serialized_spec, file)
            with spec_path.open("r", encoding="utf-8") as file:
                deserialized_spec = json.load(file)

            deserialized = deserialize_model(model_path)
            # deserialized = import_model_from_spec(deserialized_spec)

        assert isinstance(deserialized, nn.Sequential)
        assert len(deserialized) == 2
        assert isinstance(deserialized[0], nn.Linear)
        assert deserialized[0].in_features == 4
        assert deserialized[0].out_features == 8
        assert isinstance(deserialized[1], nn.ReLU)
        assert isinstance(original_weight := model[0].weight, Tensor)
        assert isinstance(deserialized_weight := deserialized[0].weight, Tensor)
        assert torch.equal(original_weight, deserialized_weight)
