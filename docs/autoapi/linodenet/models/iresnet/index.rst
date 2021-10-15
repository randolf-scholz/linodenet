


:py:mod:`linodenet.models.iresnet`
==================================

.. py:module:: linodenet.models.iresnet

.. autoapi-nested-parse::

   Implementation of invertible ResNets.









.. rubric:: Classes
.. autoapisummary::

   linodenet.models.iresnet.SpectralNorm
   linodenet.models.iresnet.LinearContraction
   linodenet.models.iresnet.iResNetBlock
   linodenet.models.iresnet.iResNet




.. rubric:: Functions
.. autoapisummary::

   linodenet.models.iresnet.spectral_norm



.. py:function:: spectral_norm(A, atol = 0.0001, rtol = 0.001, maxiter = 10)

   Compute the spectral norm `‖A‖_2` by power iteration.

   Stopping criterion:
   - maxiter reached
   - `‖ (A^TA -λI)x ‖_2 ≤ 𝗋𝗍𝗈𝗅⋅‖ λx ‖_2 + 𝖺𝗍𝗈𝗅`

   :param A:
   :type A: :class:`tensor`
   :param atol:
   :type atol: :class:`float = 1e-4`
   :param rtol:
   :type rtol: :class:`float =  1e-3,`
   :param maxiter:
   :type maxiter: :class:`int = 10`

   :returns:
   :rtype: :class:`~torch.Tensor`


.. py:class:: SpectralNorm(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`

   `‖A‖_2=λ_{𝗆𝖺𝗑}(A^𝖳A)`.

   The spectral norm `∥A∥_2 ≔ 𝗌𝗎𝗉_x ∥Ax∥_2 / ∥x∥_2` can be shown to be equal to
   `σ_\max(A) = √{λ_{𝗆𝖺𝗑} (AᵀA)}`, the largest singular value of `A`.

   It can be computed efficiently via Power iteration.

   One can show that the derivative is equal to:

   .. math::
       \frac{∂½∥A∥_2}/{∂A} = uvᵀ

   where `u,v` are the left/right-singular vector corresponding to `σ_\max`

   .. py:method:: forward(ctx, *tensors, **kwargs)
      :staticmethod:

      Forward pass.

      :param ctx:
      :param tensors:
      :param kwargs:

      :returns:
      :rtype: :class:`~torch.Tensor`


   .. py:method:: backward(ctx, *grad_outputs)
      :staticmethod:

      Backward pass.

      :param ctx:
      :param grad_outputs:



.. py:class:: LinearContraction(input_size, output_size, *, c = 0.97, bias = True)

   Bases: :py:obj:`torch.nn.Module`

   A linear layer `f(x) = A⋅x` satisfying the contraction property `‖f(x)-f(y)‖_2 ≤ ‖x-y‖_2`.

   This is achieved by normalizing the weight matrix by
   `A' = A⋅\min(\tfrac{c}{‖A‖_2}, 1)`, where `c<1` is a hyperparameter.

   :ivar input_size: The dimensionality of the input space.
   :vartype input_size: :class:`int`
   :ivar output_size: The dimensionality of the output space.
   :vartype output_size: :class:`int`
   :ivar c: The regularization hyperparameter.
   :vartype c: :class:`~torch.Tensor`
   :ivar spectral_norm: BUFFER: The value of `‖W‖_2`
   :vartype spectral_norm: :class:`~torch.Tensor`
   :ivar weight: The weight matrix.
   :vartype weight: :class:`~torch.Tensor`
   :ivar bias: The bias Tensor if present, else None.

   :vartype bias: :class:`~torch.Tensor` or :obj:`None`

   .. py:attribute:: input_size
      :annotation: :Final[int]

      

   .. py:attribute:: output_size
      :annotation: :Final[int]

      

   .. py:attribute:: c
      :annotation: :torch.Tensor

      The regularization hyperparameter.

      :type: CONST

   .. py:attribute:: one
      :annotation: :torch.Tensor

      A tensor with value 1.0

      :type: CONST

   .. py:attribute:: spectral_norm
      :annotation: :torch.Tensor

      The value of `‖W‖_2`

      :type: BUFFER

   .. py:attribute:: weight
      :annotation: :torch.Tensor

      The weight matrix.

      :type: PARAM

   .. py:attribute:: bias
      :annotation: :Optional[torch.Tensor]

      The bias term.

      :type: PARAM

   .. py:method:: reset_parameters(self)

      Reset both weight matrix and bias vector.


   .. py:method:: forward(self, x)

      Signature: `[...,n] ⟶ [...,n]`.

      :param x:
      :type x: :class:`~torch.Tensor`

      :returns:
      :rtype: :class:`~torch.Tensor`



.. py:class:: iResNetBlock(input_size, **HP)

   Bases: :py:obj:`torch.nn.Module`

   Invertible ResNet-Block of the form `g(x)=ϕ(W_1⋅W_2⋅x)`.

   By default, `W_1⋅W_2` is a low rank factorization.

   Alternative: `g(x) = W_3ϕ(W_2ϕ(W_1⋅x))`

   All linear layers must be :class:`LinearContraction` layers.
   The activation function must have Lipschitz constant `≤1` such as :class:`~torch.nn.ReLU`,
   :class:`~torch.nn.ELU` or :class:`~torch.nn.Tanh`)

   :ivar input_size: The dimensionality of the input space.
   :vartype input_size: :class:`int`
   :ivar hidden_size: The dimensionality of the latent space.
   :vartype hidden_size: :class:`int`, *default* :class:`⌊√n⌋`
   :ivar output_size: The dimensionality of the output space.
   :vartype output_size: :class:`int`
   :ivar maxiter: Maximum number of iteration in `inverse` pass
   :vartype maxiter: :class:`int`
   :ivar bottleneck: The bottleneck layers
   :vartype bottleneck: :class:`nn.Sequential`
   :ivar bias: Whether to use bias
   :vartype bias: :class:`bool`, *default* :obj:`True`
   :ivar HP: Nested dictionary containing the hyperparameters.
   :vartype HP: :class:`dict`
   :ivar residual: BUFFER: The termination error during backward propagation.
   :vartype residual: :class:`~torch.Tensor`
   :ivar bottleneck: The bottleneck layer.

   :vartype bottleneck: :class:`nn.Sequential`

   .. py:attribute:: input_size
      :annotation: :Final[int]

      The dimensionality of the inputs.

      :type: CONST

   .. py:attribute:: hidden_size
      :annotation: :Final[int]

      The dimensionality of the latents.

      :type: CONST

   .. py:attribute:: output_size
      :annotation: :Final[int]

      The dimensionality of the outputs.

      :type: CONST

   .. py:attribute:: maxiter
      :annotation: :Final[int]

      The maximum number of steps in inverse pass.

      :type: CONST

   .. py:attribute:: atol
      :annotation: :Final[float]

      The absolute tolerance threshold value.

      :type: CONST

   .. py:attribute:: rtol
      :annotation: :Final[float]

      The relative tolerance threshold value.

      :type: CONST

   .. py:attribute:: residual
      :annotation: :torch.Tensor

      The termination error during backward propagation.

      :type: BUFFER

   .. py:attribute:: HP
      :annotation: :dict

      The hyperparameter dictionary

   .. py:method:: forward(self, x)

      Signature: `[...,n] ⟶ [...,n]`.

      :param x:
      :type x: :class:`~torch.Tensor`

      :returns:
      :rtype: :class:`~torch.Tensor`


   .. py:method:: inverse(self, y)

      Compute the inverse through fixed point iteration.

      Terminates once `maxiter` or tolerance threshold
      `|x'-x|≤\text{atol} + \text{rtol}⋅|x|` is reached.

      :param y:
      :type y: :class:`~torch.Tensor`

      :returns:
      :rtype: :class:`~torch.Tensor`



.. py:class:: iResNet(input_size, **HP)

   Bases: :py:obj:`torch.nn.Module`

   Invertible ResNet consists of a stack of :class:`iResNetBlock` modules.

   :ivar input_size: The dimensionality of the input space.
   :vartype input_size: :class:`int`
   :ivar output_size: The dimensionality of the output space.
   :vartype output_size: :class:`int`
   :ivar blocks: Sequential model consisting of the iResNetBlocks
   :vartype blocks: :class:`nn.Sequential`
   :ivar reversed_blocks: The same blocks in reversed order
   :vartype reversed_blocks: :class:`nn.Sequential`
   :ivar HP: Nested dictionary containing the hyperparameters.

   :vartype HP: :class:`dict`

   .. py:attribute:: input_size
      :annotation: :Final[int]

      The dimensionality of the inputs.

      :type: CONST

   .. py:attribute:: output_size
      :annotation: :Final[int]

      The dimensionality of the outputs.

      :type: CONST

   .. py:attribute:: HP
      :annotation: :dict

      The hyperparameter dictionary

   .. py:method:: forward(self, x)

      Signature: `[...,n] ⟶ [...,n]`.

      :param x:
      :type x: :class:`~torch.Tensor`

      :returns: **xhat**
      :rtype: :class:`~torch.Tensor`


   .. py:method:: inverse(self, y)

      Compute the inverse through fix point iteration in each block in reversed order.

      :param y:
      :type y: :class:`~torch.Tensor`

      :returns: **yhat**
      :rtype: :class:`~torch.Tensor`




