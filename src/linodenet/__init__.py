r"""Linear Ordinary Differential Equation Recurrent Neural Network."""

__all__ = [
    # Constants
    "__version__",
    "CONFIG",
    # Sub-Modules
    "config",
    "initializations",
    "models",
    "projections",
    "regularizations",
    "utils",
]
import sys
from importlib import metadata
from types import ModuleType

# version check
if sys.version_info < (3, 10):
    raise RuntimeError("Python >= 3.10 required")

# pylint: disable=wrong-import-position

from linodenet import (
    config,
    initializations,
    models,
    projections,
    regularizations,
    utils,
)
from linodenet.config import CONFIG

# pylint: enable=wrong-import-position


__version__ = metadata.version(__package__)
r"""The version number of the `linodenet` package."""


# Recursively clean up namespaces to only show what the user should see.
def _clean_namespace(module: ModuleType) -> None:
    r"""Recursively cleans up the namespace.

    Sets `obj.__module__` equal to `obj.__package__` for all objects listed in
    `package.__all__` that are originating from private submodules (`package/_module.py`).
    """
    # pylint: disable=import-outside-toplevel

    from inspect import ismodule
    from logging import getLogger

    # pylint: enable=import-outside-toplevel

    assert hasattr(module, "__name__"), f"{module=} has no __name__ ?!?!"
    assert hasattr(module, "__package__"), f"{module=} has no __package__ ?!?!"
    assert hasattr(module, "__all__"), f"{module=} has no __all__!"
    assert module.__name__ == module.__package__, f"{module=} is not a package!"

    def is_private(s: str) -> bool:
        """True if starts exactly a single underscore."""
        return s.startswith("_") and not s.startswith("__")

    def is_dunder(s: str) -> bool:
        """True if starts and ends with two underscores."""
        return s.startswith("__") and s.endswith("__")

    def get_module(obj_ref: object) -> str:
        return obj_ref.__module__.rsplit(".", maxsplit=1)[-1]

    # __logger__ = logging.getLogger(module.__name__)
    # __logger__.debug("Cleaning Module!")

    module_logger = getLogger(module.__name__)
    variables = vars(module)
    maxlen = max((len(key) for key in variables))

    def _format(key: str) -> str:
        return key.ljust(maxlen)

    for key in list(variables):
        logger = module_logger.getChild(_format(key))
        obj = variables[key]

        # ignore private / dunder keys
        if is_private(key) or is_dunder(key):
            logger.debug("Skipped! - private / dunder object!")
            continue

        # special treatment for ModuleTypes
        if ismodule(obj):
            assert obj.__package__ is not None, f"{obj=} has no __package__ ?!?!"
            # subpackage!
            if obj.__package__.rsplit(".", maxsplit=1)[0] == module.__name__:
                logger.debug("Recursion!")
                _clean_namespace(obj)
            # submodule!
            elif obj.__package__ == module.__name__:
                logger.debug("Skipped! Sub-Module!")
            # 3rd party!
            else:
                logger.warning(
                    f"3rd party Module {obj.__name__!r} in {module.__name__!r}!"
                )
            continue

        if key not in module.__all__:
            logger.warning(f"Lonely Object {key!r} in {module.__name__!r}!")
        elif (isinstance(obj, type) or callable(obj)) and is_private(get_module(obj)):
            # set __module__ attribute to __package__ for functions/classes originating from private modules.
            logger.debug("Changing %s to %s!", obj.__module__, module.__package__)
            obj.__module__ = module.__package__


del sys, metadata, ModuleType
_clean_namespace(__import__(__name__))
