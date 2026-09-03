#pragma once

#include <torch/torch.h>

namespace linodenet_special {
using torch::Tensor;

autoard_bend(
    const Tensor &x,
    const Tensor &a,
    const Tensor &c,
    const Tensor &m
) - -> Tensor> Tensor;
} // namespace linodenet_special
