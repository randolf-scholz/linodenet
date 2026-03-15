#include "gaussian_transport.h"

#include <vector>

#include "hard_bend.h"
#include "ndtri_exp.h"

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::variable_list;
using torch::special::log_ndtr;

namespace {
constexpr double LOG_HALF = -0.6931471805599453;
constexpr double LOG_2PI = 1.8378770664093453;
constexpr int64_t MAXITER = 10;

void check_bimodal_args(const Tensor &x, const Tensor &mu, const Tensor &sigma) {
    TORCH_CHECK(x.is_floating_point(), "x must be a floating point tensor.");
    TORCH_CHECK(mu.is_floating_point(), "mu must be a floating point tensor.");
    TORCH_CHECK(sigma.is_floating_point(), "sigma must be a floating point tensor.");
    TORCH_CHECK(x.dtype() == mu.dtype(), "x and mu must have the same dtype.");
    TORCH_CHECK(x.dtype() == sigma.dtype(), "x and sigma must have the same dtype.");
    TORCH_CHECK(torch::all(sigma > 0).item<bool>(), "sigma must be strictly positive.");
    (void) torch::broadcast_tensors({x, mu, sigma});
}

void check_mixture_args(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) {
    TORCH_CHECK(x.is_floating_point(), "x must be a floating point tensor.");
    TORCH_CHECK(weights.is_floating_point(), "weights must be a floating point tensor.");
    TORCH_CHECK(mus.is_floating_point(), "mus must be a floating point tensor.");
    TORCH_CHECK(sigmas.is_floating_point(), "sigmas must be a floating point tensor.");
    TORCH_CHECK(weights.dtype() == x.dtype(), "weights must have the same dtype as x.");
    TORCH_CHECK(mus.dtype() == x.dtype(), "mus must have the same dtype as x.");
    TORCH_CHECK(sigmas.dtype() == x.dtype(), "sigmas must have the same dtype as x.");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1D.");
    TORCH_CHECK(mus.dim() == 1, "mus must be 1D.");
    TORCH_CHECK(sigmas.dim() == 1, "sigmas must be 1D.");
    TORCH_CHECK(
        weights.size(0) == mus.size(0) && weights.size(0) == sigmas.size(0),
        "weights, mus, and sigmas must have the same number of components."
    );
    TORCH_CHECK(torch::all(sigmas > 0).item<bool>(), "sigmas must be strictly positive.");
}

std::tuple<Tensor, Tensor, Tensor> bimodal_to_gaussian_forward_impl(
    const Tensor &x,
    const Tensor &mu,
    const Tensor &sigma
) {
    const Tensor m = mu.abs();
    const Tensor z_plus = (x + m) / sigma;
    const Tensor z_minus = (x - m) / sigma;
    const Tensor log_p = LOG_HALF + at::logaddexp(log_ndtr(z_plus), log_ndtr(z_minus));
    const Tensor log_q = LOG_HALF + at::logaddexp(log_ndtr(-z_plus),log_ndtr(-z_minus));

    Tensor y = torch::where(
        x < 0,
        linodenet_special::ndtri_exp(log_p),
        -linodenet_special::ndtri_exp(log_q)
    );
    y = y.clamp_(z_minus, z_plus);
    return {y, z_minus, z_plus};
}

Tensor bimodal_to_gaussian_x_derivative_impl(
    const Tensor &y,
    const Tensor &z_minus,
    const Tensor &z_plus,
    const Tensor &mu,
    const Tensor &sigma
) {
    const Tensor m = mu.abs();
    const Tensor y2 = y.square();
    const Tensor log_sigma = sigma.log();
    const Tensor log_phi_plus = 0.5 * (y2 - z_plus.square()) - log_sigma + LOG_HALF;
    const Tensor log_phi_minus = 0.5 * (y2 - z_minus.square()) - log_sigma + LOG_HALF;
    const Tensor d_x_exact = torch::logaddexp(log_phi_plus, log_phi_minus).exp();
    const Tensor lower_bound = torch::exp(-0.5 * (m / sigma).square()) / sigma;
    const Tensor upper_bound = sigma.reciprocal();
    return d_x_exact.clamp_( lower_bound, upper_bound);
}

std::tuple<Tensor, Tensor, Tensor> bimodal_to_gaussian_derivatives_impl(
    const Tensor &y,
    const Tensor &z_minus,
    const Tensor &z_plus,
    const Tensor &mu,
    const Tensor &sigma
) {
    const Tensor m = mu.abs();
    const Tensor mu_sign = mu.sign();
    const Tensor y2 = y.square();
    const Tensor log_sigma = sigma.log();
    const Tensor log_phi_plus = LOG_HALF + 0.5 * (y2 - z_plus.square()) - log_sigma;
    const Tensor log_phi_minus = LOG_HALF + 0.5 * (y2 - z_minus.square()) - log_sigma ;

    const Tensor d_x_exact = at::logaddexp(log_phi_plus, log_phi_minus).exp();
    const Tensor hi = torch::maximum(log_phi_plus, log_phi_minus);
    const Tensor lo = torch::minimum(log_phi_plus, log_phi_minus);
    const Tensor d_m_exact =
        torch::sign(log_phi_plus - log_phi_minus) * torch::exp(hi + torch::log1p(-torch::exp(lo - hi)));
    const Tensor d_sigma_exact =
        -(0.5 * (z_plus + z_minus) * d_x_exact + (m / sigma) * d_m_exact);

    const Tensor lower_bound = torch::exp(-0.5 * (m / sigma).square()) / sigma;
    const Tensor upper_bound = sigma.reciprocal();
    const Tensor d_x = d_x_exact.clamp(lower_bound, upper_bound);
    const Tensor d_mu = mu_sign * d_m_exact.clamp(-upper_bound, upper_bound);
    return {d_x, d_mu, d_sigma_exact};
}

Tensor gaussian_to_bimodal_guess_impl(const Tensor &x, const Tensor &mu, const Tensor &sigma) {
    const Tensor lambda = torch::exp(-0.5 * (mu / sigma).square()) / sigma;
    return linodenet_special::hard_bend(x, lambda, mu, sigma.reciprocal());
}

std::tuple<Tensor, Tensor> mixture_to_gaussian_forward_impl(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) {
    const Tensor z = (x.unsqueeze(-1) - mus) / sigmas;
    const Tensor log_w = weights.log();
    const Tensor log_p = at::logsumexp(log_w + torch::special::log_ndtr(z), -1);
    const Tensor log_q = at::logsumexp(log_w + torch::special::log_ndtr(-z), -1);

    Tensor y = torch::where(
        log_p < LOG_HALF,
        linodenet_special::ndtri_exp(log_p),
        -linodenet_special::ndtri_exp(log_q)
    );
    y = y.clamp_(std::get<0>(z.min(-1)), std::get<0>(z.max(-1)));
    return {y, z};
}

Tensor mixture_to_gaussian_x_derivative_impl(
    const Tensor &y,
    const Tensor &z,
    const Tensor &weights,
    const Tensor &sigmas
) {
    const Tensor log_ratio = 0.5 * (y.square().unsqueeze(-1) - z.square());
    const Tensor scaled_ratio = torch::exp(log_ratio + weights.log() - sigmas.log());
    return scaled_ratio.sum(-1);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> mixture_to_gaussian_derivatives_impl(
    const Tensor &y,
    const Tensor &z,
    const Tensor &weights,
    const Tensor &sigmas
) {
    const Tensor y2 = y.square();
    const Tensor log_ratio = 0.5 * (y2.unsqueeze(-1) - z.square());
    const Tensor log_w = weights.log();
    const Tensor log_sigmas = sigmas.log();
    const Tensor scaled_ratio = torch::exp(log_ratio + log_w - log_sigmas);

    const Tensor d_x = scaled_ratio.sum(-1);
    const Tensor d_mus = -scaled_ratio;
    const Tensor d_sigmas = z * -scaled_ratio;

    const Tensor log_pdf_u = -0.5 * (LOG_2PI + y2);
    const Tensor d_weights = -torch::exp(log_ndtr(-z) - log_pdf_u.unsqueeze(-1));
    return {d_x, d_weights, d_mus, d_sigmas};
}

struct BimodalToGaussian : Function<BimodalToGaussian> {
    static Tensor forward(AutogradContext *ctx, const Tensor &x, const Tensor &mu, const Tensor &sigma) {
        const auto [y, z_minus, z_plus] = bimodal_to_gaussian_forward_impl(x, mu, sigma);
        TORCH_CHECK(y.isfinite().all().item<bool>(), "bimodal_to_gaussian produced non-finite values.");
        ctx->save_for_backward({y, z_minus, z_plus, mu, sigma});
        return y;
    }

static variable_list backward(const AutogradContext *ctx, const variable_list &grad_output) {
        const auto saved = ctx->get_saved_variables();
        const Tensor &y = saved[0];
        const Tensor &z_minus = saved[1];
        const Tensor &z_plus = saved[2];
        const Tensor &mu = saved[3];
        const Tensor &sigma = saved[4];
        const Tensor &g = grad_output[0];

        const auto [d_x, d_mu, d_sigma] =
            bimodal_to_gaussian_derivatives_impl(y, z_minus, z_plus, mu, sigma);
        return {g * d_x, g * d_mu, g * d_sigma};
    }
};

struct GaussianToBimodal : Function<GaussianToBimodal> {
    static Tensor forward(AutogradContext *ctx, const Tensor &y, const Tensor &mu, const Tensor &sigma) {
        const Tensor m = mu.abs();
        Tensor lower = sigma * y - m;
        Tensor upper = sigma * y + m;
        Tensor x = gaussian_to_bimodal_guess_impl(y, mu, sigma).clamp(lower, upper);
        Tensor residual = torch::empty_like(x);
        Tensor x_newton = torch::empty_like(x);
        Tensor x_bisect = torch::empty_like(x);

        for (int64_t i = 0; i < MAXITER; ++i) {
            const auto [fx, z_minus, z_plus] = bimodal_to_gaussian_forward_impl(x, mu, sigma);
            const Tensor d_fx = bimodal_to_gaussian_x_derivative_impl(fx, z_minus, z_plus, mu, sigma);
            at::sub_out(residual, fx, y);
            at::where_out(lower, residual < 0, x, lower);
            at::where_out(upper, residual > 0, x, upper);
            at::addcdiv_out(x_newton, x, residual, d_fx, -1.0);
            at::add_out(x_bisect, lower, upper);
            at::mul_out(x_bisect, x_bisect, 0.5);
            at::where_out(x, (x_newton >= lower) & (x_newton <= upper), x_newton, x_bisect);
            x = x.clamp_(lower, upper);
        }

        x = x.clamp_(lower, upper);
        const auto [fx, z_minus, z_plus] = bimodal_to_gaussian_forward_impl(x, mu, sigma);
        ctx->save_for_backward({fx, z_minus, z_plus, mu, sigma});
        return x;
    }

    static variable_list backward(const AutogradContext *ctx, const variable_list &grad_output) {
        const auto saved = ctx->get_saved_variables();
        const Tensor &fx = saved[0];
        const Tensor &z_minus = saved[1];
        const Tensor &z_plus = saved[2];
        const Tensor &mu = saved[3];
        const Tensor &sigma = saved[4];
        const Tensor &g = grad_output[0];

        const auto [d_x, d_mu, d_sigma] =
            bimodal_to_gaussian_derivatives_impl(fx, z_minus, z_plus, mu, sigma);

        Tensor d_y = d_x.reciprocal();
        Tensor grad_mu = -d_mu * d_y;
        Tensor grad_sigma = -d_sigma * d_y;

        const Tensor upper_bound = sigma * torch::exp(0.5 * (mu / sigma).square());
        d_y = d_y.clamp_(sigma, upper_bound);
        grad_mu = grad_mu.clamp_(-upper_bound, upper_bound);

        return {g * d_y, g * grad_mu, g * grad_sigma};
    }
};

struct MixtureToGaussian : Function<MixtureToGaussian> {
    static Tensor forward(
        AutogradContext *ctx,
        const Tensor &x,
        const Tensor &weights,
        const Tensor &mus,
        const Tensor &sigmas
    ) {
        const auto [y, z] = mixture_to_gaussian_forward_impl(x, weights, mus, sigmas);
        TORCH_CHECK(y.isfinite().all().item<bool>(), "mixture_to_gaussian produced non-finite values.");
        ctx->save_for_backward({z, y, weights, sigmas});
        return y;
    }

    static variable_list backward(const AutogradContext *ctx, const variable_list &grad_output) {
        const auto saved = ctx->get_saved_variables();
        const Tensor &z = saved[0];
        const Tensor &y = saved[1];
        const Tensor &weights = saved[2];
        const Tensor &sigmas = saved[3];
        const Tensor &g = grad_output[0];

        const auto [d_values, d_weights, d_mus, d_sigmas] =
                mixture_to_gaussian_derivatives_impl(y, z, weights, sigmas);
        const Tensor grad_values = g * d_values;
        const Tensor grad_weights = at::einsum("..., ...k -> k", {g, d_weights});
        const Tensor grad_mus = at::einsum("..., ...k -> k", {g, d_mus});
        const Tensor grad_sigmas = at::einsum("..., ...k -> k", {g, d_sigmas});

        return {
            grad_values,
            // Note: project onto tangent plane.
            grad_weights - grad_weights.mean(-1, true),
            grad_mus,
            grad_sigmas
        };
    }
};

struct GaussianToMixture : Function<GaussianToMixture> {
    static Tensor forward(
        AutogradContext *ctx,
        const Tensor &y,
        const Tensor &weights,
        const Tensor &mus,
        const Tensor &sigmas
    ) {
        const Tensor lines = mus + sigmas * y.unsqueeze(-1);
        Tensor lower = std::get<0>(lines.min(-1));
        Tensor upper = std::get<0>(lines.max(-1));
        Tensor x = (weights * lines).sum(-1).clamp(lower, upper);
        Tensor residual = torch::empty_like(x);
        Tensor x_newton = torch::empty_like(x);
        Tensor x_bisect = torch::empty_like(x);

        for (int64_t i = 0; i < MAXITER; ++i) {
            const auto [fy, z] = mixture_to_gaussian_forward_impl(x, weights, mus, sigmas);
            const Tensor d_fy = mixture_to_gaussian_x_derivative_impl(fy, z, weights, sigmas);
            at::sub_out(residual, fy, y);
            at::where_out(lower, residual < 0, x, lower);
            at::where_out(upper, residual > 0, x, upper);
            at::addcdiv_out(x_newton, x, residual, d_fy, -1.0);
            at::add_out(x_bisect, lower, upper);
            at::mul_out(x_bisect, x_bisect, 0.5);
            at::where_out(x, (x_newton >= lower) & (x_newton <= upper), x_newton, x_bisect);
            x = x.clamp_(lower, upper);
        }

        const auto [fy, z] = mixture_to_gaussian_forward_impl(x, weights, mus, sigmas);
        ctx->save_for_backward({z, fy, weights, mus, sigmas});
        return x;
    }

    static variable_list backward(const AutogradContext *ctx, const variable_list &grad_output) {
        const auto saved = ctx->get_saved_variables();
        const Tensor &z = saved[0];
        const Tensor &y = saved[1];
        const Tensor &weights = saved[2];
        const Tensor &sigmas = saved[4];
        const Tensor &g = grad_output[0];

        const auto [d_x, d_weights, d_mus, d_sigmas] =
            mixture_to_gaussian_derivatives_impl(y, z, weights, sigmas);

        const Tensor grad_y = g * d_x.reciprocal();
        const Tensor grad_weights = at::einsum("..., ...k -> k", {grad_y, -d_weights});
        const Tensor grad_mus = at::einsum("..., ...k -> k", {grad_y, -d_mus});
        const Tensor grad_sigmas = at::einsum("..., ...k -> k", {grad_y, -d_sigmas});

        return {
            grad_y,
            // Note: project onto tangent plane.
            grad_weights - grad_weights.mean(-1, true),
            grad_mus,
            grad_sigmas
        };
    }
};

} // namespace

namespace linodenet_special {
Tensor bimodal_to_gaussian_meta(const Tensor &x, const Tensor &mu, const Tensor &sigma) {
    check_bimodal_args(x, mu, sigma);
    const auto tensors = torch::broadcast_tensors({x, mu, sigma});
    return torch::empty_like(tensors[0]);
}

Tensor bimodal_to_gaussian(const Tensor &x, const Tensor &mu, const Tensor &sigma) {
    return BimodalToGaussian::apply(x, mu, sigma);
}

Tensor gaussian_to_bimodal_meta(const Tensor &y, const Tensor &mu, const Tensor &sigma) {
    check_bimodal_args(y, mu, sigma);
    const auto tensors = torch::broadcast_tensors({y, mu, sigma});
    return torch::empty_like(tensors[0]);
}

Tensor gaussian_to_bimodal(const Tensor &y, const Tensor &mu, const Tensor &sigma) {
    return GaussianToBimodal::apply(y, mu, sigma);
}

Tensor mixture_to_gaussian_meta(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) {
    check_mixture_args(x, weights, mus, sigmas);
    return torch::empty_like(x);
}

Tensor mixture_to_gaussian(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) {
    return MixtureToGaussian::apply(x, weights, mus, sigmas);
}

Tensor gaussian_to_mixture_meta(
    const Tensor &y,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) {
    check_mixture_args(y, weights, mus, sigmas);
    return torch::empty_like(y);
}

Tensor gaussian_to_mixture(
    const Tensor &y,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) {
    return GaussianToMixture::apply(y, weights, mus, sigmas);
}

} // namespace linodenet_special

TORCH_LIBRARY_FRAGMENT(linodenet_special, m) {
    m.def("bimodal_to_gaussian(Tensor _, Tensor mu, Tensor sigma) -> Tensor");
    m.def("gaussian_to_bimodal(Tensor _, Tensor mu, Tensor sigma) -> Tensor");
    m.def("mixture_to_gaussian(Tensor _, Tensor weights, Tensor mus, Tensor sigmas) -> Tensor");
    m.def("gaussian_to_mixture(Tensor _, Tensor weights, Tensor mus, Tensor sigmas) -> Tensor");
}

TORCH_LIBRARY_IMPL(linodenet_special, Autograd, m) {
    m.impl("bimodal_to_gaussian", &linodenet_special::bimodal_to_gaussian);
    m.impl("gaussian_to_bimodal", &linodenet_special::gaussian_to_bimodal);
    m.impl("mixture_to_gaussian", &linodenet_special::mixture_to_gaussian);
    m.impl("gaussian_to_mixture", &linodenet_special::gaussian_to_mixture);
}

TORCH_LIBRARY_IMPL(linodenet_special, Meta, m) {
    m.impl("bimodal_to_gaussian", &linodenet_special::bimodal_to_gaussian_meta);
    m.impl("gaussian_to_bimodal", &linodenet_special::gaussian_to_bimodal_meta);
    m.impl("mixture_to_gaussian", &linodenet_special::mixture_to_gaussian_meta);
    m.impl("gaussian_to_mixture", &linodenet_special::gaussian_to_mixture_meta);
}
