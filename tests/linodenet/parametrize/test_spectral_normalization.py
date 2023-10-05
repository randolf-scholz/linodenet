"""Check that spectral norm works as a parametrization."""

import sys

import torch
from torch import nn

from linodenet.models.encoders.invertible_layers import LinearContraction
from linodenet.parametrize import SpectralNormalization

if __name__ == "__main__":
    model_a = LinearContraction(4, 4)

    print(model_a.weight, model_a.cached_weight)
    model_a.recompute_cache()
    print(model_a.weight, model_a.cached_weight)

    print("==============================================================")

    torch.manual_seed(0)

    model = nn.Linear(4, 4)
    param = nn.Parameter(model.weight.clone().detach() * 2)
    spec = SpectralNormalization(param)

    print(spec.parametrized_tensors["weight"], spec.weight, sep="\n")
    assert spec.parametrized_tensors["weight"] is param
    assert spec.weight is spec.cached_tensors["weight"]
    print("==============================================================")

    spec.cached_tensors["weight"].copy_(spec.parametrized_tensors["weight"])

    print(spec.parametrized_tensors["weight"], spec.weight, sep="\n")
    assert spec.parametrized_tensors["weight"] is param
    assert spec.weight is spec.cached_tensors["weight"]

    sys.exit(0)

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
