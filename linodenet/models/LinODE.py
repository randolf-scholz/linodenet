import torch
from torch import nn, Tensor
from typing import Union, List, Callable


class LinODECell(torch.jit.ScriptModule):
    r"""
    Linear System module

    x' = Ax + Bu + w
     y = Cx + Du + v

    """

    def __init__(self, input_size,
                 kernel_initialization: Union[torch.Tensor, Callable[int, torch.Tensor]] = None,
                 homogeneous: bool = True,
                 matrix_type: str = None,
                 device=torch.device('cpu'),
                 dtype=torch.float32,
                 ):
        r"""
        kernel_initialization: torch.tensor or callable
            either a tensor to assign to the kernel at initialization
            or a callable f: int -> torch.Tensor|L
        """
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

    @torch.jit.script_method
    def forward(self, Δt, x):
        r"""
        Inputs:
        Δt: (...,)
        x:  (..., M)

        Outputs:
        xhat:  (..., M)


        Forward using matrix exponential
        # TODO: optimize if clauses away by changing definition in constructor.
        """

        AΔt = torch.einsum('kl, ... -> ...kl', self.kernel, Δt)
        expAΔt = torch.matrix_exp(AΔt)
        xhat = torch.einsum('...kl, ...l -> ...k', expAΔt, x)

        return xhat


class LinODE(torch.jit.ScriptModule):
    r"""
    linode
    """
    def __init__(self, *cell_args, **cell_kwargs):
        r"""

        Parameters
        ----------
        cell_args
        cell_kwargs
        """
        super(LinODE, self).__init__()
        self.cell = LinODECell(*cell_args, **cell_kwargs)

    @torch.jit.script_method
    def forward(self, x0: Tensor, T: Tensor) -> Tensor:
        r"""

        Parameters
        ----------
        x0
        T

        Returns
        -------

        """
        ΔT = torch.diff(T)
        x = torch.jit.annotate(List[Tensor], [])
        x += [x0]

        for i, Δt in enumerate(ΔT):
            x += [self.cell(Δt, x[-1])]

        return torch.stack(x)




