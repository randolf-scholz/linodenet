r"""Demonstration of model configuration inference and export."""

from torch import nn

from linodenet.config import _validate_object_blueprint, initialize


class TestInitialization:
    def test_linear_model(self) -> None:
        model = nn.Linear(4, 8, bias=False)
        spec = infer_modelspec(model)
        _validate_object_blueprint(model, spec)
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
        spec = infer_modelspec(model)
        _validate_object_blueprint(model, spec)
        clone = initialize(spec)
        assert isinstance(clone, nn.Sequential)
        assert len(clone) == 2
        assert isinstance(clone[0], nn.Linear)
        assert clone[0].in_features == 4
        assert clone[0].out_features == 8
        assert isinstance(clone[1], nn.ReLU)


class TestSerialization:
    def test_linear(self) -> None:
        model = nn.Linear(4, 8, bias=False)
        spec = infer_modelspec(model)
        validate_model_spec(model, spec)
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
        spec = infer_modelspec(model)
        validate_model_spec(model, spec)

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
