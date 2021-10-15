


:py:mod:`linodenet.models`
==========================

.. py:module:: linodenet.models

.. autoapi-nested-parse::

   Models of the LinODE-Net package.





.. toctree::
   :titlesonly:
   :maxdepth: 3
   :hidden:

   iresnet/index.rst
   linodenet/index.rst

.. rubric:: Sub-Modules
.. autoapisummary::

   linodenet.models.iresnet
   linodenet.models.linodenet


.. rubric:: Attributes
.. autoapisummary::

   linodenet.models.Model
   linodenet.models.MODELS


.. rubric:: Classes
.. autoapisummary::

   linodenet.models.LinearContraction
   linodenet.models.iResNet
   linodenet.models.iResNetBlock
   linodenet.models.LinODE
   linodenet.models.LinODECell
   linodenet.models.LinODEnet






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



.. py:class:: LinODE(input_size, *, kernel_initialization = None, kernel_projection = None)

   Bases: :py:obj:`torch.nn.Module`

   Linear ODE module, to be used analogously to :func:`scipy.integrate.odeint`.

   :ivar input_size: The dimensionality of the input space.
   :vartype input_size: :class:`int`
   :ivar output_size: The dimensionality of the output space.
   :vartype output_size: :class:`int`
   :ivar kernel: The system matrix
   :vartype kernel: :class:`~torch.Tensor`
   :ivar kernel_initialization: Parameter-less function that draws a initial system matrix

   :vartype kernel_initialization: :class:`Callable[None`, :class:`Tensor]`

   .. py:attribute:: input_size
      :annotation: :Final[int]

      The dimensionality of inputs.

      :type: CONST

   .. py:attribute:: output_size
      :annotation: :Final[int]

      The dimensionality of the outputs.

      :type: CONST

   .. py:attribute:: kernel
      :annotation: :torch.Tensor

      The system matrix of the linear ODE component.

      :type: PARAM

   .. py:attribute:: xhat
      :annotation: :torch.Tensor

      The forward prediction.

      :type: BUFFER

   .. py:attribute:: kernel_initialization
      :annotation: :linodenet.initializations.Initialization

      Parameter-less function that draws a initial system matrix.

      :type: FUNC

   .. py:attribute:: kernel_projection
      :annotation: :linodenet.projections.Projection

      Regularization function for the kernel.

      :type: FUNC

   .. py:method:: forward(self, T, x0)

      Signature: `[...,N]×[...,d] ⟶ [...,N,d]`.

      :param T:
      :type T: :class:`~torch.Tensor`, :class:`shape=(...,LEN)`
      :param x0:
      :type x0: :class:`~torch.Tensor`, :class:`shape=(...,DIM)`

      :returns: **Xhat** -- The estimated true state of the system at the times `t∈T`
      :rtype: :class:`~torch.Tensor`, :class:`shape=(...,LEN,DIM)`



.. py:class:: LinODECell(input_size, *, kernel_initialization = None, kernel_projection = None)

   Bases: :py:obj:`torch.nn.Module`

   Linear System module, solves `ẋ = Ax`, i.e. `x̂ = e^{A\Delta t}x`.

   :param input_size:
   :type input_size: :class:`int`
   :param kernel_initialization:
   :type kernel_initialization: :class:`Union[Tensor`, :class:`Callable[int`, :class:`Tensor]]`

   :ivar input_size: The dimensionality of the input space.
   :vartype input_size: :class:`int`
   :ivar output_size: The dimensionality of the output space.
   :vartype output_size: :class:`int`
   :ivar kernel: The system matrix
   :vartype kernel: :class:`~torch.Tensor`
   :ivar kernel_initialization: Parameter-less function that draws a initial system matrix
   :vartype kernel_initialization: :class:`Callable[[]`, :class:`Tensor]`
   :ivar kernel_projection: Regularization function for the kernel

   :vartype kernel_projection: :class:`Callable[[Tensor]`, :class:`Tensor]`

   .. py:attribute:: input_size
      :annotation: :Final[int]

      The dimensionality of inputs.

      :type: CONST

   .. py:attribute:: output_size
      :annotation: :Final[int]

      The dimensionality of the outputs.

      :type: CONST

   .. py:attribute:: kernel
      :annotation: :torch.Tensor

      The system matrix of the linear ODE component.

      :type: PARAM

   .. py:method:: kernel_initialization(self)

      Draw an initial kernel matrix (random or static).


   .. py:method:: kernel_regularization(self, w)

      Regularize the Kernel, e.g. by projecting onto skew-symmetric matrices.


   .. py:method:: forward(self, dt, x0)

      Signature: `[...,]×[...,d] ⟶ [...,d]`.

      :param dt: The time difference `t_1 - t_0` between `x_0` and `x̂`.
      :type dt: :class:`~torch.Tensor`, :class:`shape=(...,)`
      :param x0: Time observed value at `t_0`
      :type x0: :class:`~torch.Tensor`, :class:`shape=(...,DIM)`

      :returns: **xhat** -- The predicted value at `t_1`
      :rtype: :class:`~torch.Tensor`, :class:`shape=(...,DIM)`



.. py:class:: LinODEnet(input_size, hidden_size, **HP)

   Bases: :py:obj:`torch.nn.Module`

   Linear ODE Network is a FESD model.

   +---------------------------------------------------+--------------------------------------+
   | Component                                         | Formula                              |
   +===================================================+======================================+
   | Filter  `F` (default: :class:`~torch.nn.GRUCell`) | `\hat x_i' = F(\hat x_i, x_i)`       |
   +---------------------------------------------------+--------------------------------------+
   | Encoder `ϕ` (default: :class:`~.iResNet`)         | `\hat z_i' = ϕ(\hat x_i')`           |
   +---------------------------------------------------+--------------------------------------+
   | System  `S` (default: :class:`~.LinODECell`)      | `\hat z_{i+1} = S(\hat z_i', Δ t_i)` |
   +---------------------------------------------------+--------------------------------------+
   | Decoder `π` (default: :class:`~.iResNet`)         | `\hat x_{i+1}  =  π(\hat z_{i+1})`   |
   +---------------------------------------------------+--------------------------------------+

   :ivar input_size: The dimensionality of the input space.
   :vartype input_size: :class:`int`
   :ivar hidden_size: The dimensionality of the latent space.
   :vartype hidden_size: :class:`int`
   :ivar output_size: The dimensionality of the output space.
   :vartype output_size: :class:`int`
   :ivar ZERO: BUFFER: A constant tensor of value float(0.0)
   :vartype ZERO: :class:`~torch.Tensor`
   :ivar xhat_pre: BUFFER: Stores pre-jump values.
   :vartype xhat_pre: :class:`~torch.Tensor`
   :ivar xhat_post: BUFFER: Stores post-jump values.
   :vartype xhat_post: :class:`~torch.Tensor`
   :ivar zhat_pre: BUFFER: Stores pre-jump latent values.
   :vartype zhat_pre: :class:`~torch.Tensor`
   :ivar zhat_post: BUFFER: Stores post-jump latent values.
   :vartype zhat_post: :class:`~torch.Tensor`
   :ivar kernel: PARAM: The system matrix of the linear ODE component.
   :vartype kernel: :class:`~torch.Tensor`
   :ivar encoder: MODULE: Responsible for embedding `x̂→ẑ`.
   :vartype encoder: :class:`~torch.nn.Module`
   :ivar embedding: MODULE: Responsible for embedding `x̂→ẑ`.
   :vartype embedding: :class:`~torch.nn.Module`
   :ivar system: MODULE: Responsible for propagating `ẑ_t→ẑ_{t+∆t}`.
   :vartype system: :class:`~torch.nn.Module`
   :ivar decoder: MODULE: Responsible for projecting `ẑ→x̂`.
   :vartype decoder: :class:`~torch.nn.Module`
   :ivar projection: MODULE: Responsible for projecting `ẑ→x̂`.
   :vartype projection: :class:`~torch.nn.Module`
   :ivar filter: MODULE: Responsible for updating `(x̂, x_obs) →x̂'`.

   :vartype filter: :class:`~torch.nn.Module`

   .. py:attribute:: HP
      :annotation: :dict[str, Any]

      

   .. py:attribute:: input_size
      :annotation: :Final[int]

      The dimensionality of the inputs.

      :type: CONST

   .. py:attribute:: hidden_size
      :annotation: :Final[int]

      The dimensionality of the linear ODE.

      :type: CONST

   .. py:attribute:: output_size
      :annotation: :Final[int]

      The dimensionality of the outputs.

      :type: CONST

   .. py:attribute:: concat_mask
      :annotation: :Final[bool]

      Whether to concatenate mask as extra features.

      :type: CONST

   .. py:attribute:: zero
      :annotation: :torch.Tensor

      A tensor of value float(0.0)

      :type: BUFFER

   .. py:attribute:: xhat_pre
      :annotation: :torch.Tensor

      Stores pre-jump values.

      :type: BUFFER

   .. py:attribute:: xhat_post
      :annotation: :torch.Tensor

      Stores post-jump values.

      :type: BUFFER

   .. py:attribute:: zhat_pre
      :annotation: :torch.Tensor

      Stores pre-jump latent values.

      :type: BUFFER

   .. py:attribute:: zhat_post
      :annotation: :torch.Tensor

      Stores post-jump latent values.

      :type: BUFFER

   .. py:attribute:: kernel
      :annotation: :torch.Tensor

      The system matrix of the linear ODE component.

      :type: PARAM

   .. py:method:: forward(self, T, X)

      Signature: `[...,N]×[...,N,d] ⟶ [...,N,d]`.

      **Model Sketch**::

          ⟶ [ODE] ⟶ (ẑᵢ)                (ẑᵢ') ⟶ [ODE] ⟶
                     ↓                   ↑
                    [Ψ]                 [Φ]
                     ↓                   ↑
                    (x̂ᵢ) → [ filter ] → (x̂ᵢ')
                               ↑
                            (tᵢ, xᵢ)

      :param T: The timestamps of the observations.
      :type T: :class:`~torch.Tensor`, :class:`shape=(...,LEN)` or :class:`PackedSequence`
      :param X: The observed, noisy values at times `t∈T`. Use ``NaN`` to indicate missing values.
      :type X: :class:`~torch.Tensor`, :class:`shape=(...,LEN,DIM)` or :class:`PackedSequence`

      :returns: * **X̂_pre** (:class:`~torch.Tensor`, :class:`shape=(...,LEN,DIM)`) -- The estimated true state of the system at the times `t⁻∈T` (pre-update).
                * **X̂_post** (:class:`~torch.Tensor`, :class:`shape=(...,LEN,DIM)`) -- The estimated true state of the system at the times `t⁺∈T` (post-update).

      .. rubric:: References

      - https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/



.. py:data:: Model
   

   Type hint for models.

.. py:data:: MODELS
   :annotation: :Final[dict[str, type[Model]]]

   Dictionary containing all available models.


