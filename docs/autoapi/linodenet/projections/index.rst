


:py:mod:`linodenet.projections`
===============================

.. py:module:: linodenet.projections

.. autoapi-nested-parse::

   Projection Mappings.





.. toctree::
   :titlesonly:
   :maxdepth: 3
   :hidden:

   functional/index.rst

.. rubric:: Sub-Modules
.. autoapisummary::

   linodenet.projections.functional


.. rubric:: Attributes
.. autoapisummary::

   linodenet.projections.Projection
   linodenet.projections.PROJECTIONS





.. rubric:: Functions
.. autoapisummary::

   linodenet.projections.diagonal
   linodenet.projections.identity
   linodenet.projections.normal
   linodenet.projections.orthogonal
   linodenet.projections.skew_symmetric
   linodenet.projections.symmetric



.. py:function:: diagonal(x)

   Return the closest diagonal matrix to X.

   .. math::
       \min_Y ½∥X-Y∥_F^2 s.t. Y = 𝕀⊙Y

   One can show analytically that `Y = diag(X)` is the unique minimizer.

   **Signature:** ``(..., n,n) ⟶ (..., n, n)``

   :param x:
   :type x: :class:`~torch.Tensor`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: identity(x)

   Return x as-is.

   .. math::
       \min_Y ½∥X-Y∥_F^2

   **Signature:** ``(..., n,n) ⟶ (..., n, n)``

   :param x:
   :type x: :class:`~torch.Tensor`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: normal(x)

   Return the closest normal matrix to X.

   .. math::
       \min_Y ½∥X-Y∥_F^2 s.t. YᵀY = YYᵀ

   **The Lagrangian:**

   .. math::
       ℒ(Y, Λ) = ½∥X-Y∥_F^2 + ⟨Λ, [Y, Yᵀ]⟩

   **First order necessary KKT condition:**

   .. math::
           0 &= ∇ℒ(Y, Λ) = (Y-X) + Y(Λ + Λᵀ) - (Λ + Λᵀ)Y
       \\⟺ Y &= X + [Y, Λ]

   **Second order sufficient KKT condition:**

   .. math::
            ⟨∇h|S⟩=0     &⟹ ⟨S|∇²ℒ|S⟩ ≥ 0
        \\⟺ ⟨[Y, Λ]|S⟩=0 &⟹ ⟨S|𝕀⊗𝕀 + Λ⊗𝕀 − 𝕀⊗Λ|S⟩ ≥ 0
        \\⟺ ⟨[Y, Λ]|S⟩=0 &⟹ ⟨S|S⟩ + ⟨[S, Λ]|S⟩ ≥ 0

   **Signature:** ``(..., n,n) ⟶ (..., n, n)``

   :param x:
   :type x: :class:`~torch.Tensor`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: orthogonal(x)

   Return the closest orthogonal matrix to X.

   .. math::
       \min_Y ½∥X-Y∥_F^2 s.t. Y^𝖳 Y = 𝕀 = YY^𝖳

   One can show analytically that `Y = UV^𝖳` is the unique minimizer,
   where `X=UΣV^𝖳` is the SVD of `X`.

   **Signature:** ``(..., n,n) ⟶ (..., n, n)``

   .. rubric:: References

   - <https://math.stackexchange.com/q/2215359>_

   :param x:
   :type x: :class:`~torch.Tensor`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: skew_symmetric(x)

   Return the closest skew-symmetric matrix to X.

   .. math::
       \min_Y ½∥X-Y∥_F^2 s.t. Yᵀ = -Y

   One can show analytically that Y = ½(X - Xᵀ) is the unique minimizer.

   **Signature:** ``(..., n,n) ⟶ (..., n, n)``

   :param x:
   :type x: :class:`~torch.Tensor`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:function:: symmetric(x)

   Return the closest symmetric matrix to X.

   .. math::
       \min_Y ½∥X-Y∥_F^2 s.t. Yᵀ = Y

   One can show analytically that Y = ½(X + Xᵀ) is the unique minimizer.

   **Signature:** ``(..., n,n) ⟶ (..., n, n)``

   :param x:
   :type x: :class:`~torch.Tensor`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:data:: Projection
   

   Type hint for projections.

.. py:data:: PROJECTIONS
   :annotation: :Final[dict[str, Projection]]

   Dictionary containing all available projections.


