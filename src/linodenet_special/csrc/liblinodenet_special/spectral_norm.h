#pragma once

#include <torch/torch.h>

namespace linodenet_special {
using torch::Tensor;
using torch::optional;

auto spectral_norm_meta(
    const Tensor &A,
    const optional<Tensor> &u0,
    const optional<Tensor> &v0,
    int64_t maxiter = 256,
    double atol = 1e-6,
    double rtol = 1e-6
) -> Tensor;

auto spectral_norm(
    const Tensor &A,
    const optional<Tensor> &u0,
    const optional<Tensor> &v0,
    int64_t maxiter = 256,
    double atol = 1e-6,
    double rtol = 1e-6
) -> Tensor;
} // namespace linodenet_special
