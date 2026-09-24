#pragma once

#include <torch/torch.h>

namespace imtskit_special {
using torch::Tensor;

auto hard_bend(
    const Tensor &x,
    const Tensor &a,
    const Tensor &c,
    const Tensor &m
) -> Tensor;

auto hard_bend_meta(
    const Tensor &x,
    const Tensor &a,
    const Tensor &c,
    const Tensor &m
) -> Tensor;
} // namespace imtskit_special
