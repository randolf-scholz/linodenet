#pragma once

#include <tuple>

#include <torch/torch.h>

namespace linodenet_special {
using torch::Tensor;
using torch::optional;

std::tuple<Tensor, Tensor, Tensor> singular_triplet_meta(
    const Tensor &A,
    const optional<Tensor> &u0,
    const optional<Tensor> &v0,
    int64_t maxiter = 256,
    double atol = 1e-6,
    double rtol = 1e-6
);

std::tuple<Tensor, Tensor, Tensor> singular_triplet(
    const Tensor &A,
    const optional<Tensor> &u0,
    const optional<Tensor> &v0,
    int64_t maxiter = 256,
    double atol = 1e-6,
    double rtol = 1e-6
);
}  // namespace linodenet_special
