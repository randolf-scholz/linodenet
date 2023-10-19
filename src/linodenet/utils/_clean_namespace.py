"""Clean namespace for linodenet package (unused)."""

# def clean_namespace(
#     module: ModuleType,
#     /,
#     *,
#     ignore_dunder_attributes: bool = True,
#     ignore_private_attributes: bool = True,
# ) -> bool:
#     r"""Check the namespace of a package or module.
#
#     Sets `obj.__module__` equal to `obj.__package__` for all objects listed in
#     `package.__all__` that are originating from private submodules (`package/_module.py`).
#     """
#     # check module
#     assert ismodule(module), f"{module=} is not a module!"
#     assert hasattr(module, "__name__"), f"{module=} has no __name__ ?!?!"
#     assert hasattr(module, "__package__"), f"{module=} has no __package__ ?!?!"
#     assert hasattr(module, "__all__"), f"{module=} has no __all__!!"
#     # assert (
#     #     module.__name__ == module.__package__
#     # ), f"{module.__name__=} {module.__package__=}{module=} is not a package!"
#
#     # set up logger
#     module_logger = logging.getLogger(module.__name__)
#
#     # get local variables
#     variables = vars(module)
#
#     # get max length of variable names
#     max_length = max((len(key) for key in variables))
#
#     # check all keys
#     for key, obj in variables.items():
#         # set up logger
#         logger = module_logger.getChild(key.ljust(max_length))
#
#         if ignore_dunder_attributes and is_dunder(key):
#             logger.debug("Skipped! - dunder object!")
#             continue
#         if ignore_private_attributes and is_private(key):
#             logger.debug("Skipped! - private object!")
#             continue
#         # special treatment for ModuleTypes
#         if ismodule(obj):
#             assert obj.__package__ is not None, f"{obj=} has no __package__ ?!?!"
#             # subpackage!
#             if obj.__package__.rsplit(".", maxsplit=1)[0] == module.__name__:
#                 logger.debug("Recursion!")
#                 clean_namespace(obj)
#             # submodule!
#             elif obj.__package__ == module.__name__:
#                 logger.debug("Skipped! Sub-Module!")
#             # 3rd party!
#             else:
#                 logger.warning(
#                     f"3rd party Module {obj.__name__!r} in {module.__name__!r}!"
#                 )
#             continue
#
#         if key not in module.__all__:
#             logger.warning(f"Lonely Object {key!r} in {module.__name__!r}!")
#
#         elif (isinstance(obj, type) or callable(obj)) and is_private(get_module(obj)):
#             # set __module__ attribute to __package__ for functions/classes originating from private modules.
#             logger.debug("Changing %s to %s!", obj.__module__, module.__package__)
#             assert module.__package__ is not None, f"{module=} has no __package__ ?!?!"
#             obj.__module__ = module.__package__
