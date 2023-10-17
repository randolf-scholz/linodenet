"""Check that spectral norm works as a parametrization."""


import torch
from torch import nn

from linodenet.models.encoders.invertible_layers import LinearContraction
from linodenet.parametrize import SpectralNormalization


def test_spectral_normalization() -> None:
    model_a = LinearContraction(4, 4)

    model_a.recompute_cache()

    torch.manual_seed(0)

    model = nn.Linear(4, 4)
    param = nn.Parameter(model.weight.clone().detach() * 2)
    spec = SpectralNormalization(param)

    assert spec.original_parameter is param

    # spec.cached_tensors["weight"].copy_(spec.parametrized_tensors["weight"])
    #
    # assert spec.parametrized_tensors["weight"] is param
    # assert spec.weight is spec.cached_tensors["weight"]

    # sys.exit(0)

    # print("Param ----------------------")
    # print(param, torch.linalg.cond(param), torch.linalg.matrix_norm(param, ord=2))
    #
    # weight = spec.cached_tensors["weight"]
    # print("spec.cached_tensors[weight] ----------------------")
    # print(weight, torch.linalg.cond(weight), torch.linalg.matrix_norm(weight, ord=2))
    #
    # weight = spec.parametrized_tensors["weight"]
    # print("spec.parametrized_tensors[weight] ----------------------")
    # print(weight, torch.linalg.cond(weight), torch.linalg.matrix_norm(weight, ord=2))
    #
    # weight = spec()["weight"]
    # print("spec.forward ----------------------")
    # print(spec())
    #
    # model.spec = spec
    # spec._update_cached_tensors()
    #
    # weight = spec.cached_tensors["weight"]
    # print("spec.cached_tensors[weight] ----------------------")
    # print(weight, torch.linalg.cond(weight), torch.linalg.matrix_norm(weight, ord=2))
    #
    # weight = spec()["weight"]
    # print(spec())
    # print(weight, torch.linalg.cond(weight), torch.linalg.matrix_norm(weight, ord=2))
