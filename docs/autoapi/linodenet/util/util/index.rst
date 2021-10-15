


:py:mod:`linodenet.util.util`
=============================

.. py:module:: linodenet.util.util

.. autoapi-nested-parse::

   Utility functions.







.. rubric:: Attributes
.. autoapisummary::

   linodenet.util.util.Activation
   linodenet.util.util.ACTIVATIONS





.. rubric:: Functions
.. autoapisummary::

   linodenet.util.util.deep_dict_update
   linodenet.util.util.deep_keyval_update
   linodenet.util.util.autojit
   linodenet.util.util.flatten



.. py:data:: Activation
   

   Type hint for models.

.. py:data:: ACTIVATIONS
   :annotation: :Final[dict[str, type[Activation]]]

   Dictionary containing all available activations.

.. py:function:: deep_dict_update(d, new)

   Update nested dictionary recursively in-place with new dictionary.

   Reference: https://stackoverflow.com/a/30655448/9318372

   :param d:
   :type d: :class:`dict`
   :param new:
   :type new: :class:`~collections.abc.Mapping`


.. py:function:: deep_keyval_update(d, **new_kv)

   Update nested dictionary recursively in-place with key-value pairs.

   Reference: https://stackoverflow.com/a/30655448/9318372

   :param d:
   :type d: :class:`dict`
   :param new_kv:
   :type new_kv: :class:`~collections.abc.Mapping`


.. py:function:: autojit(base_class)

   Class decorator that enables automatic jitting of nn.Modules upon instantiation.

   Makes it so that

   .. code-block:: python

       class MyModule():
           ...

       model = jit.script(MyModule())

   and

   .. code-block:: python

       @autojit
       class MyModule():
           ...

       model = MyModule()

   are (roughly?) equivalent

   :param base_class:
   :type base_class: :class:`type[nn.Module]`

   :returns:
   :rtype: :class:`type`


.. py:function:: flatten(inputs)

   Flattens element of general Hilbert space.

   :param inputs:
   :type inputs: :class:`~torch.Tensor`

   :returns:
   :rtype: :class:`~torch.Tensor`



