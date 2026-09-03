#pragma once

#include <torch/torch.h>

namespace linodenet_special {
using torch::Tensor;

auto ndtri_exp(const Tensor &log_p) -> Tensor;
} // namespace linodenet_special
