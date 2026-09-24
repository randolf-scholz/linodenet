#pragma once

#include <torch/torch.h>

namespace imtskit_special {
using torch::Tensor;

auto ndtri_exp(const Tensor &log_p) -> Tensor;
} // namespace imtskit_special
