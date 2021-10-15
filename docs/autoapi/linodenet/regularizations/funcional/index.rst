


:py:mod:`linodenet.regularizations.funcional`
=============================================

.. py:module:: linodenet.regularizations.funcional

.. autoapi-nested-parse::

   Regularizations for LinODE kernel matrix.

   Functional version.












.. rubric:: Functions
.. autoapisummary::

   linodenet.regularizations.funcional.logdetexp
   linodenet.regularizations.funcional.skew_symmetric
   linodenet.regularizations.funcional.symmetric
   linodenet.regularizations.funcional.orthogonal
   linodenet.regularizations.funcional.normal
   linodenet.regularizations.funcional.diagonal



.. py:function:: logdetexp(x, p = 1.0)

   Bias `\det(e^A)` towards 1.

   By Jacobi's formula

   .. math::
       \det(e^A) = e^{𝗍𝗋(A)} ⟺ \log(\det(e^A)) = 𝗍𝗋(A) ⟺ \log(\det(A)) = 𝗍𝗋(\log(A))

   In particular, we can regularize the LinODE model by adding a regularization term of the form

   .. math::
       |𝗍𝗋(A)|

   **Signature:** ``(..., n,n) ⟶ (...,)``

   :param x:
   :type x: :class:`~torch.Tensor`
   :param p:
   :type p: :class:`float`, *default* ``1.0``

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: skew_symmetric(x, p = None)

   Bias the matrix towards being skew-symmetric.

   **Signature:** ``(..., n,n) ⟶ (...,)``

   :param x:
   :type x: :class:`~torch.Tensor`
   :param p: If :obj:`None` uses Frobenius norm
   :type p: :class:`Optional[float]`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: symmetric(x, p = None)

   Bias the matrix towards being symmetric.

   **Signature:** ``(..., n,n) ⟶ (...,)``

   :param x:
   :type x: :class:`~torch.Tensor`
   :param p: If :obj:`None` uses Frobenius norm
   :type p: :class:`Optional[float]`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: orthogonal(x, p = None)

   Bias the matrix towards being orthogonal.

   Note that, given `n×n` matrix `X` with SVD `X=U⋅Σ⋅V^𝖳` holds

   .. math::
         &(1) &  ‖  X - ΠX‖_F &= ‖   Σ - 𝕀 ‖_F
       \\&(1) &  ‖X^𝖳 X - 𝕀‖_F &= ‖Σ^𝖳 Σ - 𝕀‖_F
       \\&(1) &  ‖X X^𝖳 - X‖_F &= ‖ΣΣ^𝖳 - 𝕀‖_F

   **Signature:** ``(..., n,n) ⟶ (...,)``

   :param x:
   :type x: :class:`~torch.Tensor`
   :param p: If :obj:`None` uses Frobenius norm
   :type p: :class:`Optional[float]`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: normal(x, p = None)

   Bias the matrix towards being normal.

   **Signature:** ``(..., n,n) ⟶ (...,)``

   :param x:
   :type x: :class:`~torch.Tensor`
   :param p: If :obj:`None` uses Frobenius norm
   :type p: :class:`Optional[float]`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: diagonal(x, p = None)

   Bias the matrix towards being diagonal.

   **Signature:** ``(..., n,n) ⟶ (...,)``

   :param x:
   :type x: :class:`~torch.Tensor`
   :param p: If :obj:`None` uses Frobenius norm
   :type p: :class:`Optional[float]`

   :returns:
   :rtype: :class:`~torch.Tensor`



