#!/usr/bin/env python
import torch
from torchinfo import summary

import linodenet.models.filters._generic
import linodenet.models.system.linode
from linodenet.models import LSSM, encoders, filters, system

if __name__ == "__main__":
    # Initialize Model
    config = {
        "input_size": 16,
        "hidden_size": 16,
        "latent_size": 16,
        "Filter": linodenet.models.filters._generic.SequentialFilter.HP,
        "System": linodenet.models.system.linode.LinODECell.HP,
        "Encoder": encoders.ResNet.HP,
        "Decoder": encoders.ResNet.HP,
    }
    model = LSSM.from_config(config)
    summary(model, depth=2)
    # Forward pass
    T = torch.linspace(0, 1, 100)
    X = torch.randn(100, 16)
    X = torch.where(X < 0, float("nan"), X)
    X_hat = model(T, X)
    # result has no missing values!
    assert not torch.isnan(X_hat).any()
