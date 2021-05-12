
LinODE-Net
================================================


.. toctree::
    :maxdepth: 3
    :caption: Contents:

    linodenet
    readme
    changelog


Example
_______

To ease the usage this package tries to follow the guidelines of scikit-learn
estimators
https://scikit-learn.org/stable/developers/develop.html. In practise the usage
looks like this:

.. code-block:: python

   import linodenet

   model = linodenet.model.LinearContraction


Installation
------------

Install the LinODE-Net_ package using ``pip`` by

.. code-block:: bash

   pip install -e .

Here we assume that you want to install the package in editable mode, because
you would like to contribute to it. This package is not available on PyPI, it
might be in the future, though.

Contribute
----------

- Issue Tracker: https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/linodenet/-/issues
- Source Code: https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/linodenet

Support
-------

If you encounter issues, please let us know.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
