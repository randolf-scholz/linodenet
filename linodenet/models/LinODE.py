import torch
from torch import nn, Tensor, jit
from typing import Union, List, Callable


class LinODECell(jit.ScriptModule):
    r"""
    Linear System module, solves $\dot x = Ax$

    **TODO:** Implement the general linear system

    .. math::
        \begin{aligned}
         \dot x &= Ax + Bu + w \\
              y &= Cx + Du + v
        \end{aligned}

    Parameters
    ----------
    input_size: int
        dimension of input
    kernel_initialization: :class:`torch.Tensor` or Callable, default=None
        Either a tensor to assign to the kernel at initialization or a callable f: :class:`int` -> :class:`torch.Tensor`
    homogeneous: bool, default=True
        Whether to include bias
    matrix_type: str, default=None
        Regularization
    device: str or :class:`torch.device`, default='cpu'
    dtype: :class:`torch.dtype`, default=:class:`torch.float32`
    """

    def __init__(self, input_size: int,
                 kernel_initialization: Union[torch.Tensor, Callable[int, torch.Tensor]] = None,
                 homogeneous: bool = True,
                 matrix_type: str = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float32,
                 ):

        super(LinODECell, self).__init__()

        if kernel_initialization is None:
            self.kernel_initialization = lambda: torch.randn(input_size, input_size) / np.sqrt(input_size)
        elif callable(kernel_initialization):
            self.kernel = lambda: torch.tensor(kernel_initialization(input_size))
        else:
            self.kernel_initialization = lambda: torch.tensor(kernel_initialization)

        self.kernel = nn.Parameter(self.kernel_initialization())

        if not homogeneous:
            self.bias = nn.Parameter(torch.randn(input_size))
            raise NotImplementedError("Inhomogeneous Linear Model not implemented yet.")

        self.to(device=device, dtype=dtype)

    @jit.script_method
    def forward(self, Δt, x):
        r"""
        Forward using matrix exponential
        # TODO: optimize if clauses away by changing definition in constructor.
        Parameters
        ----------
        Δt: :class:`torch.Tensor`
        x:  :class:`torch.Tensor`

        Returns
        -------
        xhat:  :class:`torch.Tensor`
        """

        AΔt = torch.einsum('kl, ... -> ...kl', self.kernel, Δt)
        expAΔt = torch.matrix_exp(AΔt)
        xhat = torch.einsum('...kl, ...l -> ...k', expAΔt, x)

        return xhat


class LinODE(jit.ScriptModule):
    r"""
    Linear ODE module
    """
    def __init__(self, *cell_args, **cell_kwargs):
        r"""
        Initialize Linear ODE

        Parameters
        ----------
        cell_args
        cell_kwargs
        """
        super(LinODE, self).__init__()
        self.cell = LinODECell(*cell_args, **cell_kwargs)

    @jit.script_method
    def forward(self, x0: Tensor, T: Tensor) -> Tensor:
        r"""
        Propagate x0

        Parameters
        ----------
        x0: :class:`torch.Tensor`
        T: :class:`torch.Tensor`

        Returns
        -------
        Xhat: :class:`torch.Tensor`
        """
        ΔT = torch.diff(T)
        x = torch.jit.annotate(List[Tensor], [])
        x += [x0]

        for i, Δt in enumerate(ΔT):
            x += [self.cell(Δt, x[-1])]

        return torch.stack(x)




