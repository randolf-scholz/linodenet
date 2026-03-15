#pragma once

#include <torch/torch.h>

namespace linodenet_special {
using torch::Tensor;

Tensor hard_bend_meta(
    const Tensor &x,
    const Tensor &a,
    const Tensor &c,
    const Tensor &m
);
Tensor hard_bend(
    const Tensor &x,
    const Tensor &a,
    const Tensor &c,
    const Tensor &m
);
}  // namespace linodenet_special
