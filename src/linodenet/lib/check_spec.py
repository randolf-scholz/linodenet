import logging
import sys

import torch
from torch import Tensor, jit, nn
from torch.nn.utils import parametrize as torch_parametrize

from linodenet.models.encoders.invertible_layers import LinearContraction
from linodenet.parametrize import Parametrization, SpectralNormalization
from linodenet.testing import check_model

if __name__ == "__main__":
    model = LinearContraction(4, 4)

    print(model.weight, model.cached_weight)
    model.recompute_cache()
    print(model.weight, model.cached_weight)

    print(f"==============================================================")

    torch.manual_seed(0)

    model = nn.Linear(4, 4)
    param = nn.Parameter(model.weight.clone().detach() * 2)
    spec = SpectralNormalization(param)

    print(spec.parametrized_tensors["weight"], spec.weight, sep="\n")
    assert spec.parametrized_tensors["weight"] is param
    assert spec.weight is spec.cached_tensors["weight"]
    print(f"==============================================================")

    spec.cached_tensors["weight"].copy_(spec.parametrized_tensors["weight"])

    print(spec.parametrized_tensors["weight"], spec.weight, sep="\n")
    assert spec.parametrized_tensors["weight"] is param
    assert spec.weight is spec.cached_tensors["weight"]

    sys.exit(0)

    print("Param ----------------------")
    print(param, torch.linalg.cond(param), torch.linalg.matrix_norm(param, ord=2))

    weight = spec.cached_tensors["weight"]
    print("spec.cached_tensors[weight] ----------------------")
    print(weight, torch.linalg.cond(weight), torch.linalg.matrix_norm(weight, ord=2))

    weight = spec.parametrized_tensors["weight"]
    print("spec.parametrized_tensors[weight] ----------------------")
    print(weight, torch.linalg.cond(weight), torch.linalg.matrix_norm(weight, ord=2))

    weight = spec()["weight"]
    print("spec.forward ----------------------")
    print(spec())

    model.spec = spec
    spec.recompute_cache()

    weight = spec.cached_tensors["weight"]
    print("spec.cached_tensors[weight] ----------------------")
    print(weight, torch.linalg.cond(weight), torch.linalg.matrix_norm(weight, ord=2))

    weight = spec()["weight"]
    print(spec())
    print(weight, torch.linalg.cond(weight), torch.linalg.matrix_norm(weight, ord=2))
