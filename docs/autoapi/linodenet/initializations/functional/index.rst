


:py:mod:`linodenet.initializations.functional`
==============================================

.. py:module:: linodenet.initializations.functional

.. autoapi-nested-parse::

   Initializations for the Linear ODE Networks.

   All initializations are normalized such that if `x~𝓝(0,1)`, then `Ax~𝓝(0,1)` as well.







.. rubric:: Attributes
.. autoapisummary::

   linodenet.initializations.functional.SizeLike





.. rubric:: Functions
.. autoapisummary::

   linodenet.initializations.functional.gaussian
   linodenet.initializations.functional.diagonally_dominant
   linodenet.initializations.functional.symmetric
   linodenet.initializations.functional.skew_symmetric
   linodenet.initializations.functional.orthogonal
   linodenet.initializations.functional.special_orthogonal
   linodenet.initializations.functional.canonical_skew_symmetric



.. py:data:: SizeLike
   

   Type hint for shape-like inputs.

.. py:function:: gaussian(n)

   Sample a random gaussian matrix, i.e. `A_{ij}∼𝓝(0,1/n)`.

   Normalized such that if `x∼𝓝(0,1)`, then `A⋅x∼𝓝(0,1)`

   :param n: If :class:`tuple`, the last axis is interpreted as dimension and the others as batch
   :type n: :class:`int` or :class:`tuple[int]`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: diagonally_dominant(n)

   Sample a random diagonally dominant matrix, i.e. `A = I_n + B`,with `B_{ij}∼𝓝(0,1/n²)`.

   Normalized such that if `x∼𝓝(0,1)`, then `A⋅x∼𝓝(0,1)`

   :param n: If :class:`tuple`, the last axis is interpreted as dimension and the others as batch
   :type n: :class:`int` or :class:`tuple[int]`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: symmetric(n)

   Sample a symmetric matrix, i.e. `A^⊤ = A`.

   Normalized such that if `x∼𝓝(0,1)`, then `A⋅x∼𝓝(0,1)`

   :param n:
   :type n: :class:`int` or :class:`tuple[int]`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: skew_symmetric(n)

   Sample a random skew-symmetric matrix, i.e. `A^⊤ = -A`.

   Normalized such that if `x∼𝓝(0,1)`, then `A⋅x∼𝓝(0,1)`

   :param n:
   :type n: :class:`int` or :class:`tuple[int]`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: orthogonal(n)

   Sample a random orthogonal matrix, i.e. `A^⊤ = A`.

   Normalized such that if `x∼𝓝(0,1)`, then `A⋅x∼𝓝(0,1)`

   :param n:
   :type n: :class:`int` or :class:`tuple[int]`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: special_orthogonal(n)

   Sample a random special orthogonal matrix, i.e. `A^⊤ = A^{-1}` with `\det(A)=1`.

   Normalized such that if `x∼𝓝(0,1)`, then `A⋅x∼𝓝(0,1)`

   :param n:
   :type n: :class:`int`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: canonical_skew_symmetric(n)

   Return the canonical skew symmetric matrix of size `n=2k`.

   .. math::
       𝕁_n = 𝕀_n ⊗ \begin{bmatrix}0 & +1 \\ -1 & 0\end{bmatrix}

   Normalized such that if `x∼𝓝(0,1)`, then `A⋅x∼𝓝(0,1)`

   :param n:
   :type n: :class:`int` or :class:`tuple[int]`

   :returns:
   :rtype: :class:`~torch.Tensor`



