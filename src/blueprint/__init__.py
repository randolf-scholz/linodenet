r"""Configuration protocols and utilities."""

__all__ = [
    # modules
    # "config",
    # "serialization",
    # "torch",
    # types
    "JSON",
    "Makes",
    # constants
    "BLUEPRINT_REGISTRY",
    "INFER_ARGS_REGISTRY",
    # Classes
    "BlueprintRegistry",
    "InferArgsRegistry",
    # blueprint types
    "Blueprint",
    "ModelBlueprint",
    "ObjectBlueprint",
    "TensorBlueprint",
    "is_blueprint",
    "is_model_blueprint",
    "is_tensor_blueprint",
    # protocols
    "SupportsConfig",
    "has_config",
    "is_config",
    "SupportsDefaultConfig",
    "has_default_config",
    "SupportsFromConfig",
    # functions
    "infer_args",
    "initialize",
    "infer_blueprint",
    "validate_blueprint",
    "blueprint_to_json",
]

from .config import (
    SupportsConfig,
    SupportsDefaultConfig,
    SupportsFromConfig,
    has_config,
    has_default_config,
    is_config,
)
from .core import (
    BLUEPRINT_REGISTRY,
    INFER_ARGS_REGISTRY,
    JSON,
    Blueprint,
    BlueprintRegistry,
    InferArgsRegistry,
    Makes,
    ObjectBlueprint,
    blueprint_to_json,
    infer_args,
    infer_blueprint,
    initialize,
    is_blueprint,
    validate_blueprint,
)
from .torch import (
    ModelBlueprint,
    TensorBlueprint,
    is_model_blueprint,
    is_tensor_blueprint,
)
