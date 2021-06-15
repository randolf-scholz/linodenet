r"""
Test if model init, forward and backward passes.
"""

import torch
from torch import Tensor

from linodenet.models import LinearContraction, iResNetBlock, iResNet, \
    LinODECell, LinODE, LinODEnet

# TODO: add test for cuda + add test for batch mode!

b, n, k = 20, 5, 7
X = torch.randn(b, k)
T = torch.randn(b)
ΔT = torch.diff(T)

MODELS = {
    LinearContraction: {
        'model_init'  : (k, n),
        'model_input' : (X,),
    },
    iResNetBlock: {
        'model_init'  : (k,),
        'model_input' : (X,),
    },
    iResNet: {
        'model_init'  : (k,),
        'model_input' : (X,),
    },
    LinODECell: {
        'model_init'  : (k,),
        'model_input' : (ΔT[0], X[0]),
    },
    LinODE: {
        'model_init'  : (k,),
        'model_input' : (X[0], ΔT),
    },
    LinODEnet: {
        'model_init'  : (k, 2*k),
        'model_input' : (T, X),
    },
}


def _test_model(Model: type, model_init: tuple[int, ...], model_input: tuple[Tensor, ...]):

    def err_str(s: str) -> str:
        return F"{Model=} failed {s} with {model_init=} and {model_input=}!"

    try:  # check initialization
        model = Model(*model_init)
    except Exception as E:
        raise RuntimeError(err_str("initialization")) from E

    try:  # check forward
        Xhat = model(*model_input)
    except Exception as E:
        raise RuntimeError(err_str("forward pass")) from E

    Xhat = Xhat if isinstance(Xhat, tuple) else (Xhat,)
    flat_xhat = torch.cat([xhat.flatten() for xhat in Xhat], dim=0)
    r = torch.linalg.norm(flat_xhat)

    try:  # check backward
        r.backward()
    except Exception as E:
        raise RuntimeError(err_str("backward pass")) from E


def test_all_models():
    """Checks if init, forward and backward runs for all selected models"""
    for model, params in MODELS.items():
        _test_model(model, params['model_init'], params['model_input'])
        print(F"Model {model.__name__:20s} passed!")


if __name__ == '__main__':
    test_all_models()
