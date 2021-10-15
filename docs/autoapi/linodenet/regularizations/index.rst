


:py:mod:`linodenet.regularizations`
===================================

.. py:module:: linodenet.regularizations

.. autoapi-nested-parse::

   Regularizations for LinODE kernel matrix.





.. toctree::
   :titlesonly:
   :maxdepth: 3
   :hidden:

   funcional/index.rst

.. rubric:: Sub-Modules
.. autoapisummary::

   linodenet.regularizations.funcional


.. rubric:: Attributes
.. autoapisummary::

   linodenet.regularizations.Regularization
   linodenet.regularizations.REGULARIZATIONS





.. rubric:: Functions
.. autoapisummary::

   linodenet.regularizations.diagonal
   linodenet.regularizations.logdetexp
   linodenet.regularizations.normal
   linodenet.regularizations.orthogonal
   linodenet.regularizations.skew_symmetric
   linodenet.regularizations.symmetric



.. py:function:: diagonal(x, p = None)

   Bias the matrix towards being diagonal.

   **Signature:** ``(..., n,n) ⟶ (...,)``

   :param x:
   :type x: :class:`~torch.Tensor`
   :param p: If :obj:`None` uses Frobenius norm
   :type p: :class:`Optional[float]`

   :returns:
   :rtype: :class:`~torch.Tensor`


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


.. py:function:: normal(x, p = None)

   Bias the matrix towards being normal.

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


.. py:data:: Regularization
   

   Type hint for regularizations.

.. py:data:: REGULARIZATIONS
   :annotation: :Final[dict[str, Regularization]]

   Dictionary containing all available regularizations.


