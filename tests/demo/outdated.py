# def test_export() -> None:
#     cfg = ReverseDense.DEFAULT_CONFIG(input_size=4, output_size=8)
#     export_config(asdict(cfg), Path("demo_config.json"))


### OUTDATED CODE:


def export_config(arg: object, path: FilePath) -> None:
    validate_config(arg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = path.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    def convert(value: object, prefix: str) -> object:
        match value:
            case _ if (
                is_scalar(value)
                or is_tensor_spec(value)
                or is_model_spec(value)
                or is_serialized_model_spec(value)
            ):
                return value
            case Tensor():
                return export_tensor(value, assets_dir / f"{prefix}.pt")
            case ExportedProgram():
                return export_model(value, assets_dir / f"{prefix}.pt")
            case nn.Module():
                return export_model(value, assets_dir / f"{prefix}.pt")
            case list():
                return [convert(item, f"{prefix}_{i}") for i, item in enumerate(value)]
            case tuple():
                return [convert(item, f"{prefix}_{i}") for i, item in enumerate(value)]
            case dict():
                return {k: convert(v, f"{prefix}_{k}") for k, v in value.items()}
            case _:
                raise TypeError(
                    f"Unsupported config type: {type(value).__class_name__}"
                )

    payload = convert(arg, "config")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def spec_from_dataclass(config: object) -> ModelSpec:
    assert is_dataclass(config) and not isinstance(config, type)
    model_name = config.__class__.__qualname__.split(".", 1)[0]
    model_spec: ModelSpec = {
        "__module_name__": config.__class__.__module__,
        "__class_name__": model_name,
        "__args__": [],
        "__kwargs__": asdict(config),
        "__spec_version__": SPEC_VERSION,
        "__module_version__": _infer_module_version(config.__class__.__module__),
    }
    return model_spec


def infer_spec_old[T: nn.Module](spec: Makes[T], /) -> ModelSpec:
    match spec:
        case nn.Module():
            model_spec = infer_modelspec(spec)
        case dict():
            model_spec = dict(spec)
        case type() as cls:
            model_spec = {
                "__module_name__": cls.__module__,
                "__class_name__": cls.__qualname__,
                "__args__": [],
                "__kwargs__": {},
            }
        case dtc if is_dataclass(dtc):
            model_spec = spec_from_dataclass(dtc)
        case _:
            raise TypeError(f"Unsupported model spec type: {type(spec).__class_name__}")

    if not is_model_spec(model_spec):
        raise TypeError("Expected a model spec dictionary.")

    module_name = model_spec["__module_name__"]
    class_name = model_spec["__class_name__"]
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    validate_model_spec(cls, model_spec)
    return model_spec


def validate_config(conf: object) -> None:
    match conf:
        case _ if is_scalar(conf):
            pass
        case Tensor():
            pass
        case nn.Module():
            pass
        case list() | tuple():
            for item in conf:
                validate_config(item)
        case dict():
            if (
                is_tensor_spec(conf)
                or is_model_spec(conf)
                or is_serialized_model_spec(conf)
            ):
                return
            for key, value in conf.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Config dict keys must be str, got {type(key).__class_name__}"
                    )
                if not key.isidentifier():
                    raise TypeError(f"Config dict key must be identifier: {key!r}")
                validate_config(value)
        case _:
            raise TypeError(f"Unsupported config type: {type(conf).__class_name__}")


def export_model(
    arg: nn.Module | ExportedProgram, path: FilePath
) -> SerializedModelSpec:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = infer_modelspec(arg)

    match arg:
        case torch.jit.RecursiveScriptModule():
            torch.jit.save(arg, str(path))
            fmt = "torchscript"
        case ExportedProgram():
            torch.export.save(arg, str(path))
            fmt = "torch_export"
        case nn.Module():
            torch.save(arg.state_dict(), path)
            fmt = "state_dict"
        case _:
            raise TypeError(f"Expected nn.Module, got {type(arg).__class_name__}")

    serialized_spec: SerializedModelSpec = spec | {
        "__storage_path__": str(path),
        "__storage_format__": fmt,
    }
    return serialized_spec
