import torch

from linodenet.init import INIT, gaussian, symmetric, skew_symmetric, orthogonal, special_orthogonal


def test_all_initializations():
    x = torch.normal(mean=torch.zeros(n), std=1)
