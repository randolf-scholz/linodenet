r"""Linear Ordinary Differential Equation Recurrent Neural Network."""

__all__ = [
    # Constants
    "__version__",
    "conf",
    # Sub-Modules
    "config",
    "embeddings",
    "initializations",
    "models",
    "projections",
    "regularizations",
]

import logging
from pathlib import Path
from types import ModuleType

from linodenet import (
    config,
    embeddings,
    initializations,
    models,
    projections,
    regularizations,
)
from linodenet.config import conf

__logger__ = logging.getLogger(__name__)

with open(Path(__file__).parent.joinpath("VERSION"), "r", encoding="utf8") as file:
    __version__ = file.read()
    r"""The version number of the :mod:`linodenet` package."""


# Recursively clean up namespaces to only show what the user should see.
# def clean_namespace(module: ModuleType):
#     __logger__.info(f"Cleaning {module=}")
#     variables = vars(module)
#     __logger__.info(f"Content: {list(variables)}")
#
#     assert hasattr(module, "__name__"), f"{module=} has no __name__ ?!?!"
#     assert hasattr(module, "__all__"), f"{module=} has no __all__!"
#
#     for key in list(variables):
#         __logger__.info(f"Investigating {key=} ...")
#         obj = variables[key]
#         # ignore __logger__, clean_namespace and ModuleType
#         if key in ("__logger__", "ModuleType", "clean_namespace"):
#             __logger__.info("\t skipped!")
#             continue
#         # ignore dunder keys
#         if key.startswith("__") and key.endswith("__"):
#             __logger__.info("\t skipped!")
#             continue
#         # special treatment for ModuleTypes
#         elif isinstance(obj, ModuleType):
#             # submodule!
#             if obj.__package__ == module.__name__:
#             # subpackage!
#             elif obj.__package__.rsplit(".", maxsplit=1)[0] == module.__name__:
#             # 3rd party!
#             else:
#
#             __logger__.info("\t recursion!")
#             clean_namespace(obj)
#         # delete everything not in __all__
#         if key not in module.__all__:       # type: ignore[attr-defined]
#             delattr(module, key)
#             __logger__.info("\t killed!")
#
#         # set __module__ of elements from private modules to parent module
#         elif (
#             (isinstance(obj, type) or callable(obj))
#             and module.__name__.startswith("_")
#             and not module.__name__.startswith("__")
#         ):
#             __logger__.info("\t fixing __module__!")
#             parent, _ = module.__name__.rsplit(".", maxsplit=1)
#             obj.__module__ = parent
#     else:
#         # Clean up the rest
#         for key in ("__logger__", "ModuleType", "clean_namespace"):
#             if key in variables:
#                 delattr(module, key)


#
def clean_namespace(module: ModuleType):
    assert hasattr(module, "__name__"), f"{module=} has no __name__ ?!?!"
    assert hasattr(module, "__package__"), f"{module=} has no __package__ ?!?!"
    assert hasattr(module, "__all__"), f"{module=} has no __all__!"

    # if module.__name__ != module.__package__:
    #     return

    variables = vars(module)
    for key in list(variables):
        obj = variables[key]
        # ignore clean_namespace and ModuleType
        if key in ("ModuleType", "clean_namespace"):
            continue
        # ignore dunder keys
        if key.startswith("__") and key.endswith("__"):
            continue
        # delete everything not in __all__
        if key not in module.__all__:  # type: ignore[attr-defined]
            delattr(module, key)
        # special treatment for keys in __all__
        elif isinstance(obj, ModuleType):
            clean_namespace(obj)
        # set __module__ of functions/classes from private modules to parent package
        elif (isinstance(obj, type) or callable(obj)) and module.__name__.startswith(
            "_"
        ):
            __logger__.info("\t fixing __module__!")
            parent, _ = module.__name__.rsplit(".", maxsplit=1)
            print(module.__package__)
            obj.__module__ = module.__package__

    else:
        # Clean up the rest
        for key in ("ModuleType", "clean_namespace"):
            if key in variables:
                delattr(module, key)


clean_namespace(__import__(__name__))
