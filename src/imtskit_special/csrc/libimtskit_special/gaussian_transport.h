#pragma once

#include <torch/torch.h>

namespace imtskit_special {
using torch::Tensor;

auto bimodal_to_gaussian_meta(const Tensor &x, const Tensor &mu, const Tensor &sigma) -> Tensor;

auto bimodal_to_gaussian(const Tensor &x, const Tensor &mu, const Tensor &sigma) -> Tensor;

auto bimodal_to_gaussian_value_and_grad_meta(
    const Tensor &x,
    const Tensor &mu,
    const Tensor &sigma
) -> std::tuple<Tensor, Tensor>;

auto bimodal_to_gaussian_value_and_grad(
    const Tensor &x,
    const Tensor &mu,
    const Tensor &sigma
) -> std::tuple<Tensor, Tensor>;

auto mixture_to_gaussian_meta(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) -> Tensor;

auto mixture_to_gaussian(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) -> Tensor;

auto mixture_to_gaussian_value_and_grad_meta(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) -> std::tuple<Tensor, Tensor>;

auto mixture_to_gaussian_value_and_grad(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) -> std::tuple<Tensor, Tensor>;

auto gaussian_to_bimodal_meta(
    const Tensor &y,
    const Tensor &mu,
    const Tensor &sigma,
    int64_t maxiter
) -> Tensor;

auto gaussian_to_bimodal(
    const Tensor &y,
    const Tensor &mu,
    const Tensor &sigma,
    int64_t maxiter
) -> Tensor;

auto gaussian_to_bimodal_value_and_grad_meta(
    const Tensor &y,
    const Tensor &mu,
    const Tensor &sigma,
    int64_t maxiter
) -> std::tuple<Tensor, Tensor>;

auto gaussian_to_bimodal_value_and_grad(
    const Tensor &y,
    const Tensor &mu,
    const Tensor &sigma,
    int64_t maxiter
) -> std::tuple<Tensor, Tensor>;

auto gaussian_to_mixture_meta(
    const Tensor &y,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas,
    int64_t maxiter
) -> Tensor;

auto gaussian_to_mixture(
    const Tensor &y,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas,
    int64_t maxiter
) -> Tensor;

auto gaussian_to_mixture_value_and_grad_meta(
    const Tensor &y,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas,
    int64_t maxiter
) -> std::tuple<Tensor, Tensor>;

auto gaussian_to_mixture_value_and_grad(
    const Tensor &y,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas,
    int64_t maxiter
) -> std::tuple<Tensor, Tensor>;
} // namespace imtskit_special
