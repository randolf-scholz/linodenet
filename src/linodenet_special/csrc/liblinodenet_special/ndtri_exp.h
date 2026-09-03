#pragma once

#include <torch/torch.h>

namespace linodenet_special {
using torch::Tensor;

autodtri_exp(const Tensor &log_p) - -> Tensor> Tensor;
} // namespace linodenet_special
