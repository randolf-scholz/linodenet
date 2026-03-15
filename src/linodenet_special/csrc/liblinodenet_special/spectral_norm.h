#pragma once

#include <torch/torch.h>

namespace linodenet_special {
using torch::Tensor;
using torch::optional;

Tensor spectral_norm_meta(
    const Tensor &A,
    const optional<Tensor> &u0,
    const optional<Tensor> &v0,
    const optional<int64_t> &maxiter,
    double atol = 1e-6,
    double rtol = 1e-6
);

Tensor spectral_norm(
    const Tensor &A,
    const optional<Tensor> &u0,
    const optional<Tensor> &v0,
    const optional<int64_t> &maxiter,
    double atol = 1e-6,
    double rtol = 1e-6
);
}  // namespace linodenet_special
