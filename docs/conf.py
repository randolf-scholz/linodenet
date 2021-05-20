#!/usr/bin/env python

import os
import sys
import datetime

sys.path.insert(0, os.path.abspath('../linodenet'))


# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

master_doc = "index"
project = 'LinODE-Net'
copyright = '%(year)s, %(author)s' % dict(
    year=datetime.date.today().year,
    author='Randolf Scholz'
)
author = 'Randolf Scholz'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_math_dollar',
    # 'numpydoc',
]

mathjax_config = {
    'tex2jax': {
        'inlineMath': [["\\(", "\\)"]],
        'displayMath': [["\\[", "\\]"]],
    },
}

intersphinx_mapping = {
    "matplotlib" : ("https://matplotlib.org/stable/", None),
    "numpy"      : ("https://numpy.org/doc/stable/", None),
    "python"     : ("https://docs.python.org/3/", None),
    "scipy"      : ("https://docs.scipy.org/doc/scipy/reference/", None),
    'pandas'     : ('https://pandas.pydata.org/docs', None),
    'torch'      : ('https://pytorch.org/docs/stable/', None),
}

autodoc_default_options = {
    'members':         True,
    'member-order':    'bysource',
    'special-members': '__init__',
}

# generate autosummary even if no references
# autosummary_generate = True
# autoclass_content = 'both'

# autodoc_mock_imports = ["torch"]    # https://github.com/sphinx-doc/sphinx/issues/6521#issuecomment-505765893

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_typehints = 'none'

# recognizes custom types
autodoc_type_aliases = {
    "Tensor": "torch.Tensor",
    "tensor": "torch.Tensor",
    # "Tensor": r":class:`~torch.Tensor`",
    # "tensor": r":class:`~torch.Tensor`",
    "iResNet": r"class:`~linodenet.models.iResNet",
}

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False
autodoc_docstring_signature = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# --  sphinx.ext.napoleon configuration ---------------------------------------
# True to use the :ivar: role for instance variables. False to use the .. attribute:: directive instead.
# Defaults to False.
napoleon_use_ivar = True

# True to use a :param: role for each function parameter.
# False to use a single :parameters: role for all the parameters.
napoleon_use_param = True       # Defaults to True.
# True to use a :keyword: role for each function keyword argument.
# False to use a single :keyword arguments: role for all the keywords.
napoleon_use_keyword = True    # Defaults to True.
# True to use the :rtype: role for the return type.
# False to output the return type inline with the description.
napoleon_use_rtype = False      # Defaults to True.
# A mapping to translate type names to other names or references.
# Works only when napoleon_use_param = True. Defaults to None.
napoleon_type_aliases = {
    "Tensor": r":class:`~torch.Tensor`",
    "tensor" : r":class:`~torch.Tensor`",
    "iResNet" : r"class:`~linodenet.models.iResNet",
}
# True to allow using PEP 526 attributes annotations in classes. If an attribute is documented in the docstring without
# a type and has an annotation in the class body, that type is used.
napoleon_attr_annotations = True

# -- sphinx_autodoc_typehints configuration ----------------------------------

# # set_type_checking_flag (default: False): if True, set typing.TYPE_CHECKING to True to enable "expensive" typing
# # imports
# set_type_checking_flag = False
# # typehints_fully_qualified (default: False): if True, class names are always fully qualified (e.g.
# # module.for.Class). If False, just the class name displays (e.g. Class)
# typehints_fully_qualified = False
# # always_document_param_types (default: False): If False, do not add type info for undocumented parameters. If True,
# # add stub documentation for undocumented parameters to be able to add type info.
# always_document_param_types = False
# # typehints_document_rtype (default: True): If False, never add an :rtype: directive. If True, add the :rtype:
# # directive if no existing :rtype: is found.
# typehints_document_rtype = True
# # simplify_optional_unions (default: True): If True, optional parameters of type "Union[...]" are simplified as being
# # of type Union[..., None] in the resulting documentation (e.g. Optional[Union[A, B]] -> Union[A, B, None]). If False,
# # the "Optional"-type is kept. Note: If False, any Union containing None will be displayed as Optional! Note: If an
# # optional parameter has only a single type (e.g Optional[A] or Union[A, None]), it will always be displayed as
# # Optional!
# simplify_optional_unions = True


# -- numpydoc configuration ----------------------------------

# numpydoc_show_inherited_class_members = False
# # numpydoc_attributes_as_param_list = False
# numpydoc_xref_param_type = True
#
# numpydoc_xref_aliases = {
#     "Tensor": r"torch.Tensor",
#     "tensor": r"torch.Tensor",
#     "iResNet": r"class:`~linodenet.models.iResNet",
# }
