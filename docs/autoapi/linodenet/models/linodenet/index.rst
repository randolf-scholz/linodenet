


:py:mod:`linodenet.models.linodenet`
====================================

.. py:module:: linodenet.models.linodenet

.. autoapi-nested-parse::

   Contains implementations of ODE models.









.. rubric:: Classes
.. autoapisummary::

   linodenet.models.linodenet.LinODECell
   linodenet.models.linodenet.LinODE
   linodenet.models.linodenet.LinODEnet






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




