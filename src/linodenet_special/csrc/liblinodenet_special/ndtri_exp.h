#pragma once

#include <torch/torch.h>

namespace linodenet_special {
using torch::Tensor;

Tensor ndtri_exp(const Tensor &log_p);
}  // namespace linodenet_special
