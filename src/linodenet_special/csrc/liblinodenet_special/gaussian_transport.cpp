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

void check_bimodal_args(const Tensor &x, const Tensor &mu, const Tensor &sigma) {
    TORCH_CHECK(x.is_floating_point(), "x must be a floating point tensor.");
    TORCH_CHECK(mu.is_floating_point(), "mu must be a floating point tensor.");
    TORCH_CHECK(sigma.is_floating_point(), "sigma must be a floating point tensor.");
    TORCH_CHECK(x.dtype() == mu.dtype(), "x and mu must have the same dtype.");
    TORCH_CHECK(x.dtype() == sigma.dtype(), "x and sigma must have the same dtype.");
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
}

std::tuple<Tensor, Tensor, Tensor> bimodal_value_and_stats(
    const Tensor &x,
    const Tensor &mu,
    const Tensor &sigma
) {
    const Tensor mu_abs = mu.abs();
    const Tensor z_plus = (x + mu_abs) / sigma;
    const Tensor z_minus = (x - mu_abs) / sigma;
    const Tensor log_p = LOG_HALF + at::logaddexp(log_ndtr(z_plus), log_ndtr(z_minus));
    const Tensor log_q = LOG_HALF + at::logaddexp(log_ndtr(-z_plus),log_ndtr(-z_minus));
    // Switch between lower-tail and upper-tail evaluations to avoid cancellation near 0 and 1.
    Tensor y = torch::where(
        log_p < LOG_HALF,
        linodenet_special::ndtri_exp(log_p),
        -linodenet_special::ndtri_exp(log_q)
    );
    return {y.clamp_(z_minus, z_plus), z_plus, z_minus};
}

std::tuple<Tensor, Tensor> bimodal_to_gaussian_value_and_grad(
    const Tensor &x,
    const Tensor &mu,
    const Tensor &sigma
) {
    const auto [fx, z_plus, z_minus] = bimodal_value_and_stats(x, mu, sigma);
    const Tensor log_sigma = sigma.log();
    const Tensor lower_bound = exp(-0.5 * (mu / sigma).square()) / sigma;
    const Tensor upper_bound = sigma.reciprocal();
    const Tensor y2 = fx.square();
    const Tensor log_phi_plus = 0.5 * (y2 - z_plus.square()) - log_sigma + LOG_HALF;
    const Tensor log_phi_minus = 0.5 * (y2 - z_minus.square()) - log_sigma + LOG_HALF;
    const Tensor d_fx = exp(logaddexp(log_phi_plus, log_phi_minus)).clamp_( lower_bound, upper_bound);

    return {fx, d_fx};
}


/*
 * \dv{y}{x} &= ½σ⁻¹(ℯ^{½(y²-z₊²)} + ℯ^{½(y²-z₋²)}) \\
 * \dv{y}{μ} &= ½σ⁻¹(ℯ^{½(y²-z₊²)} - ℯ^{½(y²-z₋²)}) \\
 * \dv{y}{σ} &= -½σ⁻¹(z₊ℯ^{½(y²-z₊²)} + z₋ℯ^{½(y²-z₋²)})
 */
std::tuple<Tensor, Tensor, Tensor> bimodal_to_gaussian_derivatives(
    const Tensor &x,
    const Tensor &mu,
    const Tensor &sigma,
    const Tensor &y
) {
    const Tensor mu_abs = mu.abs();
    const Tensor z_plus = (x + mu_abs) / sigma;
    const Tensor z_minus = (x - mu_abs) / sigma;
    const Tensor log_sigma = sigma.log();
    const Tensor mu_sign = mu.sign();
    const Tensor y2 = y.square();
    // Evaluate the two mode contributions in log space to avoid tail underflow.
    const Tensor log_phi_plus = LOG_HALF + 0.5 * (y2 - z_plus.square()) - log_sigma;
    const Tensor log_phi_minus = LOG_HALF + 0.5 * (y2 - z_minus.square()) - log_sigma ;
    const Tensor hi = maximum(log_phi_plus, log_phi_minus);
    const Tensor lo = minimum(log_phi_plus, log_phi_minus);

    Tensor d_x = exp(logaddexp(log_phi_plus, log_phi_minus));
    Tensor d_mu_abs = sign(log_phi_plus - log_phi_minus) * exp(hi + log1p(-exp(lo - hi)));
    Tensor d_sigma = -(x * d_x + mu_abs * d_mu_abs) / sigma;

    // The analytic slope lives in [exp(-½(m/σ)²)/σ, 1/σ]; clamp only to absorb drift.
    const Tensor lower_bound = exp(-0.5 * (mu_abs / sigma).square()) / sigma;
    const Tensor upper_bound = sigma.reciprocal();
    d_x = d_x.clamp_(lower_bound, upper_bound);
    Tensor d_mu = mu_sign * d_mu_abs.clamp_(-upper_bound, upper_bound);

    return {d_x, d_mu, d_sigma};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> bimodal_to_gaussian_derivatives2(
    const Tensor &x,
    const Tensor &mu,
    const Tensor &sigma,
    const Tensor &y
) {
    const Tensor mu_abs = mu.abs();
    const Tensor z_plus = (x + mu_abs) / sigma;
    const Tensor z_minus = (x - mu_abs) / sigma;
    const Tensor log_sigma = sigma.log();
    const Tensor mu_sign = mu.sign();
    const Tensor y2 = y.square();
    const Tensor log_phi_plus = 0.5 * (y2 - z_plus.square()) - log_sigma + LOG_HALF;
    const Tensor log_phi_minus = 0.5 * (y2 - z_minus.square()) - log_sigma + LOG_HALF;
    const Tensor log_norm = at::logaddexp(log_phi_plus, log_phi_minus);

    Tensor d_x = log_norm.exp();
    const Tensor w_plus = (log_phi_plus - log_norm).exp();
    const Tensor w_minus = (log_phi_minus - log_norm).exp();
    Tensor d_mu_abs = d_x * (w_plus - w_minus);
    const Tensor d_sigma_exact =
        -(0.5 * (z_plus + z_minus) * d_x + (mu_abs / sigma) * d_mu_abs);

    const Tensor lower_bound = (-0.5 * (mu_abs / sigma).square()).exp() / sigma;
    const Tensor upper_bound = sigma.reciprocal();
    d_x = d_x.clamp_(lower_bound, upper_bound);
    const Tensor d_mu = mu_sign * d_mu_abs.clamp_(-upper_bound, upper_bound);

    // Reuse d_x as the common scale so only the normalized two-mode weights
    // need to be carried into the second-derivative terms.
    const Tensor z_avg = z_plus * w_plus + z_minus * w_minus;
    const Tensor z_diff = z_plus * w_plus - z_minus * w_minus;
    const Tensor z2_avg = z_plus.square() * w_plus + z_minus.square() * w_minus;
    const Tensor z_term_sum = d_x * z_avg / sigma;
    const Tensor z_term_diff = d_x * z_diff / sigma;
    const Tensor z2_term_sum = d_x * z2_avg / sigma;

    const Tensor d2_x = y * d_x.square() - z_term_sum;
    const Tensor d2_mu = mu_sign * (y * d_x * d_mu_abs - z_term_diff);
    const Tensor d2_sigma = y * d_x * d_sigma_exact - d_x / sigma + z2_term_sum;

    return {d_x, d_mu, d_sigma_exact, d2_x, d2_mu, d2_sigma};
}

Tensor gaussian_to_bimodal_guess(const Tensor &x, const Tensor &mu, const Tensor &sigma) {
    // Match the slope at the origin with the hard-bend surrogate to get a cheap
    // initial guess for the safeguarded Newton iteration.
    const Tensor lambda = exp(-0.5 * (mu / sigma).square()) / sigma;
    return linodenet_special::hard_bend(x, lambda, mu, sigma.reciprocal());
}

Tensor gaussian_to_bimodal_value(
    const Tensor &y,
    const Tensor &mu,
    const Tensor &sigma,
    const int64_t maxiter
) {
    const Tensor m = mu.abs();
    Tensor lower = sigma * y - m;
    Tensor upper = sigma * y + m;
    Tensor x = gaussian_to_bimodal_guess(y, mu, sigma);

    for (int64_t i = 0; i < maxiter; ++i) {
        x = x.clamp_(lower, upper);
        const auto [fx, d_fx] = bimodal_to_gaussian_value_and_grad(x, mu, sigma);
        const Tensor residual = fx - y;
        // Since T is monotone, the sign of the residual tells us which side of the
        // bracket still contains the inverse solution.
        lower = torch::where(residual < 0, x, lower);
        upper = torch::where(residual > 0, x, upper);
        const Tensor x_newton = x - residual / d_fx;
        const Tensor x_bisect = 0.5 * (lower + upper);
        x = torch::where((x_newton >= lower) & (x_newton <= upper), x_newton, x_bisect);
    }

    return x.clamp_(lower, upper);
}

std::tuple<Tensor, Tensor, Tensor> mixture_value_and_stats(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) {
    const Tensor z = (x.unsqueeze(-1) - mus) / sigmas;
    const Tensor log_w = weights.log();
    const Tensor log_p = logsumexp(log_w + log_ndtr(z), -1);
    const Tensor log_q = logsumexp(log_w + log_ndtr(-z), -1);
    const Tensor lower = std::get<0>(z.min(-1));
    const Tensor upper = std::get<0>(z.max(-1));
    // Switch between lower-tail and upper-tail evaluations to avoid cancellation near 0 and 1.
    Tensor y = where(
        log_p < LOG_HALF,
        linodenet_special::ndtri_exp(log_p),
        -linodenet_special::ndtri_exp(log_q)
    ).clamp_(lower, upper);

    return {y, z, log_w};
}

/*
 * ∂y/∂x &= ∑ₖ (ωₖ/σₖ) ℯ^{½(y²-zₖ²)}, \\
 * ∂y/∂ωₖ &= \sqrt{2π} ℯ^{½y²}(Φ(zₖ) - (1/n)∑ⱼΦ(zⱼ)), \\
 * ∂y/∂μₖ &= -(ωₖ/σₖ) ℯ^{½(y²-zₖ²)}, \\
 * ∂y/∂σₖ &= -(ωₖ zₖ/σₖ) ℯ^{½(y²-zₖ²)}.
 */
std::tuple<Tensor, Tensor, Tensor, Tensor> mixture_to_gaussian_derivatives(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas,
    const Tensor &y
) {
    const Tensor z = (x.unsqueeze(-1) - mus) / sigmas;
    const Tensor log_w = weights.log();
    const Tensor log_sigmas = sigmas.log();
    const Tensor y2 = y.square();
    // exp(½(y² - zₖ²)) = φ(zₖ) / φ(y)
    const Tensor log_ratio = 0.5 * (y2.unsqueeze(-1) - z.square());
    // (ωₖ / σₖ) exp(½(y² - zₖ²)) appears in ∂y/∂x, ∂y/∂μₖ, and ∂y/∂σₖ.
    const Tensor scaled_ratio = exp(log_ratio + log_w - log_sigmas);

    const Tensor d_x = scaled_ratio.sum(-1);
    const Tensor d_mus = -scaled_ratio;
    const Tensor d_sigmas = z * -scaled_ratio;

    // ∂y/∂ωₖ = √(2π) ℯ^{½y²}⋅(Φ(zₖ) - (1/n)∑ⱼΦ(zⱼ)).
    // Factor out max(log Φ(zₖ)) to keep the centered CDF difference in a stable range.
    const Tensor log_pdf_u = (-0.5 * (LOG_2PI + y2)).unsqueeze(-1);
    const Tensor log_phi = log_ndtr(z);
    const Tensor log_phi_max = std::get<0>(log_phi.max(-1, true));
    const Tensor scaled_phi = exp(log_phi - log_phi_max);
    const Tensor centered_scaled_phi = scaled_phi - scaled_phi.mean(-1, true);
    const Tensor d_weights = exp(log_phi_max - log_pdf_u) * centered_scaled_phi;

    return {d_x, d_weights, d_mus, d_sigmas};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
mixture_to_gaussian_derivatives2(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas,
    const Tensor &y
) {
    const auto [_, z, log_w] = mixture_value_and_stats(x, weights, mus, sigmas);
    const Tensor log_sigmas = sigmas.log();
    const Tensor y2 = y.square();
    const Tensor log_ratio = 0.5 * (y2.unsqueeze(-1) - z.square());
    const Tensor scaled_ratio = (log_ratio + log_w - log_sigmas).exp();

    const Tensor d_x = scaled_ratio.sum(-1);
    const Tensor d_mus = -scaled_ratio;
    const Tensor d_sigmas = -z * scaled_ratio;

    const Tensor log_pdf_u = (-0.5 * (LOG_2PI + y2)).unsqueeze(-1);
    const Tensor log_phi = log_ndtr(z);
    const Tensor log_phi_max = std::get<0>(log_phi.max(-1, true));
    const Tensor scaled_phi = (log_phi - log_phi_max).exp();
    const Tensor centered_scaled_phi = scaled_phi - scaled_phi.mean(-1, true);
    const Tensor d_weights = (log_phi_max - log_pdf_u).exp() * centered_scaled_phi;

    // Eₖ / σₖ reappears in multiple mixed second derivatives, so keep that
    // common factor grouped before forming the final Jacobian entries.
    const Tensor e_over_sigma = (log_ratio - log_sigmas).exp();
    const Tensor d2_x = y * d_x.square() + (d_sigmas / sigmas).sum(-1);
    const Tensor d2_weights =
        y.unsqueeze(-1) * d_x.unsqueeze(-1) * d_weights
        + e_over_sigma
        - e_over_sigma.mean(-1, true);
    const Tensor d2_mus =
        y.unsqueeze(-1) * d_x.unsqueeze(-1) * d_mus - d_sigmas / sigmas;
    const Tensor d2_sigmas =
        y.unsqueeze(-1) * d_x.unsqueeze(-1) * d_sigmas
        + (z.square() - 1) * scaled_ratio / sigmas;

    return {d_x, d_weights, d_mus, d_sigmas, d2_x, d2_weights, d2_mus, d2_sigmas};
}

std::tuple<Tensor, Tensor> mixture_to_gaussian_value_and_grad(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) {
    const auto [fx, z, log_w] = mixture_value_and_stats(x, weights, mus, sigmas);
    const Tensor log_sigmas = sigmas.log();
    const Tensor log_ratio = 0.5 * (fx.square().unsqueeze(-1) - z.square());
    const Tensor d_fx = exp(log_ratio + log_w - log_sigmas).sum(-1);
    return {fx, d_fx};
}

Tensor gaussian_to_mixture_value(
    const Tensor &y,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas,
    const int64_t maxiter
) {
    // Each component alone would invert y to xₖ = μₖ + σₖy. The mixture inverse
    // must lie between the smallest and largest of these affine tail candidates,
    // so we use their pointwise min/max as a safe bracket and their weighted mean
    // as a cheap initial guess for the safeguarded Newton iteration.
    const Tensor lines = mus + sigmas * y.unsqueeze(-1);
    Tensor lower = std::get<0>(lines.min(-1));
    Tensor upper = std::get<0>(lines.max(-1));
    Tensor x = torch::linalg_vecdot(weights, lines, -1);

    for (int64_t i = 0; i < maxiter; ++i) {
        x = x.clamp_(lower, upper);
        const auto [fx, d_fx] = mixture_to_gaussian_value_and_grad(x, weights, mus, sigmas);
        const Tensor residual = fx - y;
        // Since T is monotone, the sign of the residual tells us which side of the
        // bracket still contains the inverse solution.
        lower = torch::where(residual < 0, x, lower);
        upper = torch::where(residual > 0, x, upper);
        const Tensor x_newton = x - residual / d_fx;
        const Tensor x_bisect = 0.5 * (lower + upper);
        x = torch::where((x_newton >= lower) & (x_newton <= upper), x_newton, x_bisect);
    }

    return x.clamp_(lower, upper);
}


struct BimodalToGaussian : Function<BimodalToGaussian> {
    static Tensor forward(AutogradContext *ctx, const Tensor &x, const Tensor &mu, const Tensor &sigma) {
        torch::NoGradGuard guard;
        const Tensor y = std::get<0>(bimodal_value_and_stats(x, mu, sigma));
        ctx->save_for_backward({x, mu, sigma, y});
        return y;
    }

    static variable_list backward(
        const AutogradContext *ctx,
        const variable_list &grad_output
    ) {
        const Tensor &g = grad_output[0];
        const auto saved = ctx->get_saved_variables();
        const Tensor &x = saved[0];
        const Tensor &mu = saved[1];
        const Tensor &sigma = saved[2];
        const Tensor &y = saved[3];

        const auto [d_x, d_mu, d_sigma] =
            bimodal_to_gaussian_derivatives(x, mu, sigma, y);
        return {g * d_x, g * d_mu, g * d_sigma};
    }
};

struct BimodalToGaussianValueAndGrad : Function<BimodalToGaussianValueAndGrad> {
    static variable_list forward(
        AutogradContext *ctx,
        const Tensor &x,
        const Tensor &mu,
        const Tensor &sigma
    ) {
        torch::NoGradGuard guard;
        const auto [y, d_x] = bimodal_to_gaussian_value_and_grad(x, mu, sigma);
        ctx->save_for_backward({x, mu, sigma, y});
        return {y, d_x};
    }

    static variable_list backward(
        const AutogradContext *ctx,
        const variable_list &grad_output
    ) {
        const Tensor &grad_y = grad_output[0];
        const Tensor &grad_dy = grad_output[1];
        const auto saved = ctx->get_saved_variables();
        const Tensor &x = saved[0];
        const Tensor &mu = saved[1];
        const Tensor &sigma = saved[2];
        const Tensor &y = saved[3];

        const auto [d_x, d_mu, d_sigma, d2_x, d2_mu, d2_sigma] =
            bimodal_to_gaussian_derivatives2(x, mu, sigma, y);
        return {
            grad_y * d_x + grad_dy * d2_x,
            grad_y * d_mu + grad_dy * d2_mu,
            grad_y * d_sigma + grad_dy * d2_sigma,
        };
    }
};

struct GaussianToBimodal : Function<GaussianToBimodal> {
    static Tensor forward(
        AutogradContext *ctx,
        const Tensor &y,
        const Tensor &mu,
        const Tensor &sigma,
        const int64_t maxiter
    ) {
        torch::NoGradGuard guard;
        const Tensor x = gaussian_to_bimodal_value(y, mu, sigma, maxiter);
        const Tensor fx = std::get<0>(bimodal_value_and_stats(x, mu, sigma));
        ctx->save_for_backward({x,mu, sigma, fx});
        return x;
    }

    static variable_list backward(
        const AutogradContext *ctx,
        const variable_list &grad_output
    ) {
        const auto saved = ctx->get_saved_variables();
        const Tensor &x = saved[0];
        const Tensor &mu = saved[1];
        const Tensor &sigma = saved[2];
        const Tensor &y = saved[3];
        const Tensor &g = grad_output[0];

        const auto [d_x, d_mu, d_sigma] =
            bimodal_to_gaussian_derivatives(x, mu, sigma, y);

        Tensor d_y = d_x.reciprocal();
        Tensor grad_mu = -d_mu * d_y;
        Tensor grad_sigma = -d_sigma * d_y;

        const Tensor upper_bound = sigma * exp(0.5 * (mu / sigma).square());
        d_y = d_y.clamp_(sigma, upper_bound);
        grad_mu = grad_mu.clamp_(-1, +1);

        return {g * d_y, g * grad_mu, g * grad_sigma, Tensor()};
    }
};

struct GaussianToBimodalValueAndGrad : Function<GaussianToBimodalValueAndGrad> {
    static variable_list forward(
        AutogradContext *ctx,
        const Tensor &y,
        const Tensor &mu,
        const Tensor &sigma,
        const int64_t maxiter
    ) {
        torch::NoGradGuard guard;
        const Tensor x = gaussian_to_bimodal_value(y, mu, sigma, maxiter);
        const auto [fx, df_x] = bimodal_to_gaussian_value_and_grad(x, mu, sigma);
        ctx->save_for_backward({x, mu, sigma, fx});
        return {x, df_x.reciprocal()};
    }

    static variable_list backward(
        const AutogradContext *ctx,
        const variable_list &grad_output
    ) {
        const Tensor &grad_x = grad_output[0];
        const Tensor &grad_dx = grad_output[1];
        const auto saved = ctx->get_saved_variables();
        const Tensor &x = saved[0];
        const Tensor &mu = saved[1];
        const Tensor &sigma = saved[2];
        const Tensor &fx = saved[3];

        const auto [d_x, d_mu, d_sigma, d2_x, d2_mu, d2_sigma] =
            bimodal_to_gaussian_derivatives2(x, mu, sigma, fx);
        const Tensor dx_inv = d_x.reciprocal();

        Tensor d_y = dx_inv;
        Tensor d_mu_inv = -d_mu * dx_inv;
        const Tensor d_sigma_inv = -d_sigma * dx_inv;

        const Tensor upper_bound = sigma * (0.5 * (mu / sigma).square()).exp();
        d_y = d_y.clamp_(sigma, upper_bound);
        d_mu_inv = d_mu_inv.clamp_(-1, 1);

        // j = ∂x/∂y = (∂T/∂x)⁻¹, so differentiating the inverse map once more
        // introduces the cubic power of dx_inv in these Jacobian-output terms.
        const Tensor dx_inv3 = dx_inv.pow(3);
        const Tensor j_y = -d2_x * dx_inv3;
        const Tensor j_mu = (d2_x * d_mu - d2_mu * d_x) * dx_inv3;
        const Tensor j_sigma = (d2_x * d_sigma - d2_sigma * d_x) * dx_inv3;

        return {
            grad_x * d_y + grad_dx * j_y,
            grad_x * d_mu_inv + grad_dx * j_mu,
            grad_x * d_sigma_inv + grad_dx * j_sigma,
            Tensor()
        };
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
        torch::NoGradGuard guard;
        const Tensor y = std::get<0>(mixture_value_and_stats(x, weights, mus, sigmas));
        ctx->save_for_backward({x, weights, mus, sigmas, y});
        return y;
    }

    static variable_list backward(
        const AutogradContext *ctx,
        const variable_list &grad_output
    ) {
        const Tensor &g = grad_output[0];
        const auto saved = ctx->get_saved_variables();
        const Tensor &x = saved[0];
        const Tensor &weights = saved[1];
        const Tensor &mus = saved[2];
        const Tensor &sigmas = saved[3];
        const Tensor &y = saved[4];

        const auto [d_values, d_weights, d_mus, d_sigmas] =
            mixture_to_gaussian_derivatives(x, weights, mus, sigmas, y);
        return {
            g * d_values,
            g.unsqueeze(-1) * d_weights,
            g.unsqueeze(-1) * d_mus,
            g.unsqueeze(-1) * d_sigmas,
        };
    }
};

struct MixtureToGaussianValueAndGrad : Function<MixtureToGaussianValueAndGrad> {
    static variable_list forward(
        AutogradContext *ctx,
        const Tensor &x,
        const Tensor &weights,
        const Tensor &mus,
        const Tensor &sigmas
    ) {
        torch::NoGradGuard guard;
        const auto [y, d_x] = mixture_to_gaussian_value_and_grad(x, weights, mus, sigmas);
        ctx->save_for_backward({x, weights, mus, sigmas, y});
        return {y, d_x};
    }

    static variable_list backward(
        const AutogradContext *ctx,
        const variable_list &grad_output
    ) {
        const Tensor &grad_y = grad_output[0];
        const Tensor &grad_dy = grad_output[1];
        const auto saved = ctx->get_saved_variables();
        const Tensor &x = saved[0];
        const Tensor &weights = saved[1];
        const Tensor &mus = saved[2];
        const Tensor &sigmas = saved[3];
        const Tensor &y = saved[4];

        const auto [d_x, d_weights, d_mus, d_sigmas, d2_x, d2_weights, d2_mus, d2_sigmas] =
            mixture_to_gaussian_derivatives2(x, weights, mus, sigmas, y);
        return {
            grad_y * d_x + grad_dy * d2_x,
            grad_y.unsqueeze(-1) * d_weights + grad_dy.unsqueeze(-1) * d2_weights,
            grad_y.unsqueeze(-1) * d_mus + grad_dy.unsqueeze(-1) * d2_mus,
            grad_y.unsqueeze(-1) * d_sigmas + grad_dy.unsqueeze(-1) * d2_sigmas,
        };
    }
};

struct GaussianToMixture : Function<GaussianToMixture> {
    static Tensor forward(
        AutogradContext *ctx,
        const Tensor &y,
        const Tensor &weights,
        const Tensor &mus,
        const Tensor &sigmas,
        const int64_t maxiter
    ) {
        torch::NoGradGuard guard;
        const Tensor x = gaussian_to_mixture_value(y, weights, mus, sigmas, maxiter);
        const Tensor fx = std::get<0>(mixture_value_and_stats(x, weights, mus, sigmas));
        ctx->save_for_backward({x, weights, mus, sigmas, fx});
        return x;
    }

    static variable_list backward(
        const AutogradContext *ctx,
        const variable_list &grad_output
    ) {
        const Tensor &g = grad_output[0];
        const auto saved = ctx->get_saved_variables();
        const Tensor &x = saved[0];
        const Tensor &weights = saved[1];
        const Tensor &mus = saved[2];
        const Tensor &sigmas = saved[3];
        const Tensor &y = saved[4];

        const auto [d_x, d_weights, d_mus, d_sigmas] =
            mixture_to_gaussian_derivatives(x, weights, mus, sigmas, y);

        const Tensor grad_y = g * d_x.reciprocal();
        const Tensor outer_grad = -grad_y.unsqueeze(-1);

        return {
            grad_y,
            outer_grad * d_weights,
            outer_grad * d_mus,
            outer_grad * d_sigmas,
            Tensor()
        };
    }
};

struct GaussianToMixtureValueAndGrad : Function<GaussianToMixtureValueAndGrad> {
    static variable_list forward(
        AutogradContext *ctx,
        const Tensor &y,
        const Tensor &weights,
        const Tensor &mus,
        const Tensor &sigmas,
        const int64_t maxiter
    ) {
        torch::NoGradGuard guard;
        const Tensor x = gaussian_to_mixture_value(y, weights, mus, sigmas, maxiter);
        const auto [y_star, dy_star] = mixture_to_gaussian_value_and_grad(x, weights, mus, sigmas);
        ctx->save_for_backward({x, weights, mus, sigmas, y_star});
        return {x, dy_star.reciprocal()};
    }

    static variable_list backward(
        const AutogradContext *ctx,
        const variable_list &grad_output
    ) {
        const Tensor &grad_x = grad_output[0];
        const Tensor &grad_dx = grad_output[1];
        const auto saved = ctx->get_saved_variables();
        const Tensor &x = saved[0];
        const Tensor &weights = saved[1];
        const Tensor &mus = saved[2];
        const Tensor &sigmas = saved[3];
        const Tensor &y = saved[4];

        const auto [d_x, d_weights, d_mus, d_sigmas, d2_x, d2_weights, d2_mus, d2_sigmas] =
            mixture_to_gaussian_derivatives2(x, weights, mus, sigmas, y);
        const Tensor dx_inv = d_x.reciprocal();
        // Every inverse-map parameter derivative is -(∂T/∂θ)/(∂T/∂x), and the
        // Jacobian-output derivatives reuse the same dx_inv³ factor.
        const Tensor dx_inv3 = dx_inv.pow(3);

        const Tensor d_y = dx_inv;
        const Tensor d_weights_inv = -d_weights * dx_inv.unsqueeze(-1);
        const Tensor d_mus_inv = -d_mus * dx_inv.unsqueeze(-1);
        const Tensor d_sigmas_inv = -d_sigmas * dx_inv.unsqueeze(-1);

        const Tensor j_y = -d2_x * dx_inv3;
        const Tensor j_weights =
            (d2_x.unsqueeze(-1) * d_weights - d2_weights * d_x.unsqueeze(-1))
            * dx_inv3.unsqueeze(-1);
        const Tensor j_mus =
            (d2_x.unsqueeze(-1) * d_mus - d2_mus * d_x.unsqueeze(-1))
            * dx_inv3.unsqueeze(-1);
        const Tensor j_sigmas =
            (d2_x.unsqueeze(-1) * d_sigmas - d2_sigmas * d_x.unsqueeze(-1))
            * dx_inv3.unsqueeze(-1);

        return {
            grad_x * d_y + grad_dx * j_y,
            grad_x.unsqueeze(-1) * d_weights_inv + grad_dx.unsqueeze(-1) * j_weights,
            grad_x.unsqueeze(-1) * d_mus_inv + grad_dx.unsqueeze(-1) * j_mus,
            grad_x.unsqueeze(-1) * d_sigmas_inv + grad_dx.unsqueeze(-1) * j_sigmas,
            Tensor()
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

std::tuple<Tensor, Tensor> bimodal_to_gaussian_value_and_grad_meta(
    const Tensor &x,
    const Tensor &mu,
    const Tensor &sigma
) {
    const Tensor y = bimodal_to_gaussian_meta(x, mu, sigma);
    return {y, torch::empty_like(y)};
}

Tensor gaussian_to_bimodal_meta(
    const Tensor &y,
    const Tensor &mu,
    const Tensor &sigma,
    const int64_t maxiter
) {
    check_bimodal_args(y, mu, sigma);
    TORCH_CHECK(maxiter >= 0, "maxiter must be a non-negative integer.");
    const auto tensors = torch::broadcast_tensors({y, mu, sigma});
    return torch::empty_like(tensors[0]);
}

std::tuple<Tensor, Tensor> gaussian_to_bimodal_value_and_grad_meta(
    const Tensor &y,
    const Tensor &mu,
    const Tensor &sigma,
    const int64_t maxiter
) {
    const Tensor x = gaussian_to_bimodal_meta(y, mu, sigma, maxiter);
    return {x, torch::empty_like(x)};
}

Tensor bimodal_to_gaussian(const Tensor &x, const Tensor &mu, const Tensor &sigma) {
    return BimodalToGaussian::apply(x, mu, sigma);
}

std::tuple<Tensor, Tensor> bimodal_to_gaussian_value_and_grad(
    const Tensor &x,
    const Tensor &mu,
    const Tensor &sigma
) {
    auto output = BimodalToGaussianValueAndGrad::apply(x, mu, sigma);
    return {output[0], output[1]};
}

Tensor gaussian_to_bimodal(
    const Tensor &y,
    const Tensor &mu,
    const Tensor &sigma,
    const int64_t maxiter
) {
    return GaussianToBimodal::apply(y, mu, sigma, maxiter);
}

std::tuple<Tensor, Tensor> gaussian_to_bimodal_value_and_grad(
    const Tensor &y,
    const Tensor &mu,
    const Tensor &sigma,
    const int64_t maxiter
) {
    auto output = GaussianToBimodalValueAndGrad::apply(y, mu, sigma, maxiter);
    return {output[0], output[1]};
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

std::tuple<Tensor, Tensor> mixture_to_gaussian_value_and_grad_meta(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) {
    const Tensor y = mixture_to_gaussian_meta(x, weights, mus, sigmas);
    return {y, torch::empty_like(y)};
}

Tensor gaussian_to_mixture_meta(
    const Tensor &y,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas,
    const int64_t maxiter
) {
    check_mixture_args(y, weights, mus, sigmas);
    TORCH_CHECK(maxiter >= 0, "maxiter must be a non-negative integer.");
    return torch::empty_like(y);
}

std::tuple<Tensor, Tensor> gaussian_to_mixture_value_and_grad_meta(
    const Tensor &y,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas,
    const int64_t maxiter
) {
    const Tensor x = gaussian_to_mixture_meta(y, weights, mus, sigmas, maxiter);
    return {x, torch::empty_like(x)};
}

Tensor mixture_to_gaussian(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) {
    return MixtureToGaussian::apply(x, weights, mus, sigmas);
}

std::tuple<Tensor, Tensor> mixture_to_gaussian_value_and_grad(
    const Tensor &x,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas
) {
    auto output = MixtureToGaussianValueAndGrad::apply(x, weights, mus, sigmas);
    return {output[0], output[1]};
}

Tensor gaussian_to_mixture(
    const Tensor &y,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas,
    const int64_t maxiter
) {
    return GaussianToMixture::apply(y, weights, mus, sigmas, maxiter);
}

std::tuple<Tensor, Tensor> gaussian_to_mixture_value_and_grad(
    const Tensor &y,
    const Tensor &weights,
    const Tensor &mus,
    const Tensor &sigmas,
    const int64_t maxiter
) {
    auto output = GaussianToMixtureValueAndGrad::apply(y, weights, mus, sigmas, maxiter);
    return {output[0], output[1]};
}

} // namespace linodenet_special

TORCH_LIBRARY_FRAGMENT(linodenet_special, m) {
    m.def("bimodal_to_gaussian(Tensor _, Tensor mu, Tensor sigma) -> Tensor");
    m.def("gaussian_to_bimodal(Tensor _, Tensor mu, Tensor sigma, int maxiter) -> Tensor");
    m.def("mixture_to_gaussian(Tensor _, Tensor weights, Tensor mus, Tensor sigmas) -> Tensor");
    m.def("gaussian_to_mixture(Tensor _, Tensor weights, Tensor mus, Tensor sigmas, int maxiter) -> Tensor");
    m.def("bimodal_to_gaussian_value_and_grad(Tensor _, Tensor mu, Tensor sigma) -> (Tensor, Tensor)");
    m.def("gaussian_to_bimodal_value_and_grad(Tensor _, Tensor mu, Tensor sigma, int maxiter) -> (Tensor, Tensor)");
    m.def("mixture_to_gaussian_value_and_grad(Tensor _, Tensor weights, Tensor mus, Tensor sigmas) -> (Tensor, Tensor)");
    m.def("gaussian_to_mixture_value_and_grad(Tensor _, Tensor weights, Tensor mus, Tensor sigmas, int maxiter) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(linodenet_special, Autograd, m) {
    m.impl("bimodal_to_gaussian", &linodenet_special::bimodal_to_gaussian);
    m.impl("gaussian_to_bimodal", &linodenet_special::gaussian_to_bimodal);
    m.impl("mixture_to_gaussian", &linodenet_special::mixture_to_gaussian);
    m.impl("gaussian_to_mixture", &linodenet_special::gaussian_to_mixture);
    m.impl("bimodal_to_gaussian_value_and_grad", &linodenet_special::bimodal_to_gaussian_value_and_grad);
    m.impl("gaussian_to_bimodal_value_and_grad", &linodenet_special::gaussian_to_bimodal_value_and_grad);
    m.impl("mixture_to_gaussian_value_and_grad", &linodenet_special::mixture_to_gaussian_value_and_grad);
    m.impl("gaussian_to_mixture_value_and_grad", &linodenet_special::gaussian_to_mixture_value_and_grad);
}

TORCH_LIBRARY_IMPL(linodenet_special, Meta, m) {
    m.impl("bimodal_to_gaussian", &linodenet_special::bimodal_to_gaussian_meta);
    m.impl("gaussian_to_bimodal", &linodenet_special::gaussian_to_bimodal_meta);
    m.impl("mixture_to_gaussian", &linodenet_special::mixture_to_gaussian_meta);
    m.impl("gaussian_to_mixture", &linodenet_special::gaussian_to_mixture_meta);
    m.impl("bimodal_to_gaussian_value_and_grad", &linodenet_special::bimodal_to_gaussian_value_and_grad_meta);
    m.impl("gaussian_to_bimodal_value_and_grad", &linodenet_special::gaussian_to_bimodal_value_and_grad_meta);
    m.impl("mixture_to_gaussian_value_and_grad", &linodenet_special::mixture_to_gaussian_value_and_grad_meta);
    m.impl("gaussian_to_mixture_value_and_grad", &linodenet_special::gaussian_to_mixture_value_and_grad_meta);
}
