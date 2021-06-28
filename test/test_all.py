r"""
Test if model init, forward and backward passes.
"""

import logging
from itertools import product

import torch
from torch import Tensor

from linodenet.models import LinearContraction, iResNetBlock, iResNet, LinODECell, LinODE, LinODEnet

logger = logging.getLogger(__name__)

OUTER_BATCH = 3
INNER_BATCH = 5
LEN = 9  # sequence LENgth
DIM = 7  # input DIMension
OUT = 6  # OUTput dimension
LAT = 8  # LATent dimension

DTYPE = torch.float32
DEVICES = [torch.device('cpu')]

if torch.cuda.is_available():
    DEVICES.append(torch.device('cuda'))

BATCH_SHAPES = [(), (INNER_BATCH,), (OUTER_BATCH, INNER_BATCH)]

MODELS = {
    LinearContraction: {
        'initialization' : (DIM, OUT),
        'input_shapes'   : ((LEN, DIM),),  # X
        'output_shapes'  : ((LEN, OUT),),
    },
    iResNetBlock: {
        'initialization' : (DIM, ),
        'input_shapes'   : ((LEN, DIM),),  # X
        'output_shapes'  : ((LEN, DIM),),
    },
    iResNet: {
        'initialization' : (DIM,),
        'input_shapes'   : ((LEN, DIM),),  # X
        'output_shapes'  : ((LEN, DIM),),
    },
    LinODECell: {
        'initialization' : (DIM,),
        'input_shapes'   : ((), (DIM, )),  # Δt, x0
        'output_shapes'  : ((DIM, ),),
    },
    LinODE: {
        'initialization' : (DIM,),
        'input_shapes'   : ((LEN, ), (DIM,)),  # T, x0
        'output_shapes'  : ((LEN, DIM),),
    },
    LinODEnet: {
        'initialization' : (DIM, LAT),
        'input_shapes'   : ((LEN,), (LEN, DIM)),  # T, X
        'output_shapes'  : ((LEN, DIM),),
    },
}


def _make_tensors(shapes: tuple[tuple[int, ...]], batch_sizes: tuple[int] = (),
                  dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')
                  ) -> tuple[Tensor]:
    """Makes tensors of required shape with potentially multiple batch dimensions added to all tensors"""
    tensors = []
    for shape in shapes:
        batched_shape = (*batch_sizes, *shape)
        tensor = torch.randn(batched_shape, dtype=dtype, device=device)
        tensors.append(tensor)

    return tuple(tensors)


def _test_model(Model: type, initialization: tuple[int, ...],
                inputs: tuple[Tensor, ...], targets: tuple[Tensor, ...],
                device: torch.device = torch.device('cpu')):

    def err_str(s: str) -> str:
        return F"{Model=} failed {s} with {initialization=} and {inputs=}!"

    logger.info(">>> INITIALIZATION with %s", initialization)
    try:  # check initialization
        model = Model(*initialization)
        model.to(dtype=DTYPE, device=device)
    except Exception as E:
        raise RuntimeError(err_str("initialization")) from E
    logger.info(">>> INITIALIZATION \N{HEAVY CHECK MARK}")

    logger.info(">>> FORWARD with input shapes %s", [tuple(x.shape) for x in inputs])
    try:  # check forward
        outputs = model(*inputs)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    except Exception as E:
        raise RuntimeError(err_str("forward pass")) from E

    assert all(output.shape == target.shape for output, target in zip(outputs, targets))
    logger.info(">>> Output shapes %s match with targets!", [tuple(x.shape) for x in targets])
    logger.info(">>> FORWARD \N{HEAVY CHECK MARK}")

    logger.info(">>> BACKWARD TEST")
    loss = sum([torch.mean((output - target)**2) for output, target in zip(outputs, targets)])
    try:  # check backward
        loss.backward()
    except Exception as E:
        raise RuntimeError(err_str("backward pass")) from E
    logger.info(">>> BACKWARD \N{HEAVY CHECK MARK}")


def test_all_models():
    """Checks if init, forward and backward runs for all selected models"""
    for model, params in MODELS.items():
        logger.info("Testing %s", model.__name__)
        initialization = params['initialization']
        input_shapes   = params['input_shapes']
        output_shapes  = params['output_shapes']

        for device, batch_shape in product(DEVICES, BATCH_SHAPES):
            logger.info("Testing %s on %s with batch_shape %s", model.__name__, device, batch_shape)
            inputs  = _make_tensors(input_shapes, batch_shape, dtype=DTYPE, device=device)
            targets = _make_tensors(output_shapes, batch_shape, dtype=DTYPE, device=device)
            _test_model(model, initialization, inputs, targets, device=device)

        logger.info("Model %s passed all tests!!", model.__name__)


if __name__ == '__main__':
    test_all_models()
