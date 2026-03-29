#pragma once

#include <torch/torch.h>

namespace linodenet_special {
using torch::Tensor;

Tensor bimodal_to_gaussian_meta(const Tensor &x, const Tensor &mu, const Tensor &sigma);
Tensor bimodal_to_gaussian(const Tensor &x, const Tensor &mu, const Tensor &sigma);

Tensor mixture_to_gaussian_meta(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
);
Tensor mixture_to_gaussian(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
);

Tensor gaussian_to_bimodal_meta(
    const Tensor &y,
    const Tensor &mu,
    const Tensor &sigma,
    int64_t maxiter
);
Tensor gaussian_to_bimodal(
    const Tensor &y,
    const Tensor &mu,
    const Tensor &sigma,
    int64_t maxiter
);

Tensor gaussian_to_mixture_meta(
    const Tensor &y,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas,
    int64_t maxiter
);
Tensor gaussian_to_mixture(
    const Tensor &y,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas,
    int64_t maxiter
);
}  // namespace linodenet_special
