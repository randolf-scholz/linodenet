


:py:mod:`linodenet.embeddings`
==============================

.. py:module:: linodenet.embeddings

.. autoapi-nested-parse::

   #TODO add module summary line.

   #TODO add module description.





.. toctree::
   :titlesonly:
   :maxdepth: 3
   :hidden:

   modules/index.rst

.. rubric:: Sub-Modules
.. autoapisummary::

   linodenet.embeddings.modules




.. rubric:: Classes
.. autoapisummary::

   linodenet.embeddings.ConcatEmbedding
   linodenet.embeddings.ConcatProjection






.. py:class:: ConcatEmbedding(input_size, hidden_size)

   Bases: :py:obj:`torch.nn.Module`

   Maps `x ⟼ [x,w]`.

   :ivar input_size: 
   :vartype input_size: :class:`int`
   :ivar hidden_size: 
   :vartype hidden_size: :class:`int`
   :ivar pad_size: 
   :vartype pad_size: :class:`int`
   :ivar padding:
   :vartype padding: :class:`~torch.Tensor`

   .. py:attribute:: input_size
      :annotation: :Final[int]

      The dimensionality of the inputs.

      :type: CONST

   .. py:attribute:: hidden_size
      :annotation: :Final[int]

      The dimensionality of the outputs.

      :type: CONST

   .. py:attribute:: pad_size
      :annotation: :Final[int]

      The size of the padding.

      :type: CONST

   .. py:attribute:: padding
      :annotation: :torch.Tensor

      The padding vector.

      :type: PARAM

   .. py:method:: forward(self, X)

      Signature: `[..., d] ⟶ [..., d+e]`.

      :param X:
      :type X: :class:`~torch.Tensor`, :class:`shape=(...,DIM)`

      :returns:
      :rtype: :class:`~torch.Tensor`, :class:`shape=(...,LAT)`


   .. py:method:: inverse(self, Z)

      Signature: `[..., d+e] ⟶ [..., d]`.

      The reverse of the forward. Satisfies inverse(forward(x)) = x for any input.

      :param Z:
      :type Z: :class:`~torch.Tensor`, :class:`shape=(...,LEN,LAT)`

      :returns:
      :rtype: :class:`~torch.Tensor`, :class:`shape=(...,LEN,DIM)`



.. py:class:: ConcatProjection(input_size, hidden_size)

   Bases: :py:obj:`torch.nn.Module`

   Maps `z = [x,w] ⟼ x`.

   :ivar input_size: 
   :vartype input_size: :class:`int`
   :ivar hidden_size:
   :vartype hidden_size: :class:`int`

   .. py:attribute:: input_size
      :annotation: :Final[int]

      The dimensionality of the inputs.

      :type: CONST

   .. py:attribute:: hidden_size
      :annotation: :Final[int]

      The dimensionality of the outputs.

      :type: CONST

   .. py:method:: forward(self, Z)

      Signature: `[..., d+e] ⟶ [..., d]`.

      :param Z:
      :type Z: :class:`~torch.Tensor`, :class:`shape=(...,LEN,LAT)`

      :returns:
      :rtype: :class:`~torch.Tensor`, :class:`shape=(...,LEN,DIM)`




