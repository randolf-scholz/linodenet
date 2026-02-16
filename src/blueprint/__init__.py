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
    # blueprint types
    "Blueprint",
    "InferArgsRegistry",
    "ObjectBlueprint",
    "ModelBlueprint",
    "TensorBlueprint",
    "BlueprintRegistry",
    "is_blueprint",
    "is_model_blueprint",
    "blueprint_to_json",
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
    "_initialize_from_args",
    "infer_blueprint",
    "validate_blueprint",
]

from blueprint.config import (
    SupportsConfig,
    SupportsDefaultConfig,
    SupportsFromConfig,
    has_config,
    has_default_config,
    is_config,
)
from blueprint.core import (
    BLUEPRINT_REGISTRY,
    INFER_ARGS_REGISTRY,
    JSON,
    Blueprint,
    BlueprintRegistry,
    InferArgsRegistry,
    Makes,
    ObjectBlueprint,
    _initialize_from_args,
    blueprint_to_json,
    infer_args,
    infer_blueprint,
    initialize,
    is_blueprint,
    validate_blueprint,
)
from blueprint.torch import ModelBlueprint, TensorBlueprint, is_model_blueprint
