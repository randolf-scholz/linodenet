// #include <ATen/ATen.h>
#include "spectral_norm.h"

// #include <torch/linalg.h>
// #include <vector>
// #include <string>

// import someLib as sl      ⟶  namespace sl = someLib;
// from someLib import func  ⟶  using someLib::func;
// from someLib import *     ⟶  using namespace someLib;
using torch::optional;
using torch::Tensor;
using torch::outer;
using torch::dot;
using torch::autograd::variable_list;
using torch::autograd::AutogradContext;
using torch::autograd::Function;

namespace linodenet_special {

/*
 * NOTE: discontinuity of singular vectors.
 *
 * A = [[1, 0], [0, 1+ε]]
 * Then σ₁ = 1, σ₂ = 1+ε;
 * right singular vectors are [1, 0] and [0, 1]
 * left singular vectors are [1, 0] and [0, 1]
 * singular dyads are [1, 0]⊗[1, 0]= [[1,0],[0,0]] and [0, 1]⊗[0, 1]= [[0,0],[0,1]]
 *
 * B = [[1, ε],[ε,1]]
 * Then σ₁ = 1+ε, σ₂ = 1-ε;
 * right singular vectors are [1, 1] and [1, -1] (un-normalized)
 * left singular vectors are [1, 1] and [1, -1] (un-normalized)
 * singular dyads are [1, 1]⊗[1, 1]= [[1,1],[1,1]] and [1, -1]⊗[1, -1]= [[1,-1],[-1,1]]
 *
 * Therefore, the singular dyads are discontinuous in the matrix entries.
 * This happens when singular values are repeated.
 * Since every path from A to B requires a singular value to be repeated, this is a general problem.
 * However, there should be "good" paths, that, in a path connected sense deform the singular dyads.
 * Case in point: when A is the identity matrix, then **every** vector is a singular vector.
 *
 * */

 /*
  * IDEA: regularized rank-1 approximation
  * argmin_{u,v} ‖A - uvᵀ‖² + λ‖u‖²‖v‖²
  *   in this case, R_λ(u,v) = ‖u‖²‖v‖² is the resulting singular value
  *   the larger λ, the smaller the discovered σ. However, we want σ ≥ σₘₐₓ.
  */


struct SpectralNorm: public Function<SpectralNorm> {
    /** @brief Spectral norm of a matrix.
     *
     * Formalizing as a optimization problem:
     * By Eckard-Young Theorem: min_{u,v} ‖A - σuvᵀ‖² s.t. ‖u‖₂ = ‖v‖₂ = 1
     * Equivalently: max_{u,v} ⟨A∣uvᵀ⟩ s.t. ‖u‖₂ = ‖v‖₂ = 1
     *
     * This is a non-convex QCQP, in standard form:
     * max_{(u,v)}  ½ [u, v]ᵀ [[0, A], [Aᵀ, 0]] [u, v]
     * s.t. [u, v]ᵀ [[𝕀ₘ, 0], [0, 0]] [u, v] - 1 =0
     * and  [u, v]ᵀ [[0, 0], [0, 𝕀ₙ]] [u, v] - 1 =0
     *
     * @ref https://math.stackexchange.com/questions/4658991
     * @ref https://math.stackexchange.com/questions/4697688
     *
     * Lagrangian: L(u,v,λ,μ) = uᵀAv - λ(uᵀu - 1) - μ(vᵀv - 1)
     * KKT conditions: ∇L = 0 ⟺ A v - 2λu = 0 ⟺ [-2λ𝕀ₘ, A    ] [u] = [0]
     *                          Aᵀu - 2μv = 0   [Aᵀ   , -2μ𝕀ₙ] [v] = [0]
     *
     * Second order conditions:  sᵀ∇²Ls ≥ 0 uf ∇hᵀs = 0
     * ∇hᵀ = [2uᵀ, 2vᵀ]
     * ∇²L =  [-2λ𝕀ₘ, A    ]
     *        [Aᵀ   , -2μ𝕀ₙ]
     *
     * NOTE: the gradient is linear, and the problem is a quadratic optimization problem!
     * in particular, the problem can be solved by a single Newton step!
     *
     * Equality constrained optimization problem:
     * The first order convergence criterion is ‖Av-σu‖₂ = 0 and ‖Aᵀu-σv‖₂ = 0
     * Plugging in the iteration, we get ‖u' - σũ‖ = 0 and ‖v' - σṽ‖ = 0 (tilde indicates normalized vector)
     * secondly we can estimate σ in each iteration via one of the 3 formulas
     * (1) σ = uᵀAv  (2) σᵤ = ũᵀu'  (3) σᵥ = ṽᵀv'
     * Plugging these into the equations we get
     * ‖u' -  u'ᵀ ũᵀũ‖
     * Error estimate: Note that
     * ‖Av - σu‖ = ‖σ̃ũ - σu‖ = ‖σ̃ũ - σũ + σũ -σu‖ ≤ ‖σ̃ũ - σũ‖ + ‖σũ -σu‖ = (σ̃ - σ) + σ‖ũ - u‖
     *
     * @note (Stopping criterion):
     *     The standard stopping criterion for a non-negative smooth function is
     *     ‖∇f(x)‖ ≤ α + β⋅f(x)
     *
     *     Here, we factorize into two parts for u and v respectively:
     *
     *     ‖∇ᵤf(u,v)‖ ≤ α + β⋅f(u,v) and ‖∇ᵥf(u,v)‖ ≤ α + β⋅f(u,v)
     *
     *     iff
     *
     *     ‖Av - σu‖ ≤ α + β⋅σ and ‖Aᵀu - σv‖ ≤ α + β⋅σ
     *
     *     iff, using ũ = Av and ṽ=Aᵀu, u'= ũ/‖ũ‖ and v'=  ṽ/‖ṽ‖ and σ = ⟨u'∣u⟩ = ⟨v'∣v⟩
     *
     *     ‖ũ - σu‖ ≤ α + β⋅σ and ‖ṽ - σv‖ ≤ α + β⋅σ
     *
     * @note (Alt. stopping criterion):
     *     Plugging in the definition of ũ and σ, and dividing by ‖ũ‖ yields, using u'=  ũ/‖ũ‖
     *
     *     ‖u'-⟨u∣u'⟩u‖ ≤ α/‖ũ‖ + β ⟨u∣u'⟩
     *
     *     close to convergence, ⟨u∣u'⟩ ≈ 1, giving the stopping criterion
     *
     *     ‖u'-u‖ ≤ α/‖ũ‖ + β
     *
     *     Assuming ‖ũ‖>1, we can the first term. Squaring gives the final criterion:i
     *
     *     ‖u'-u‖² ≤ β²
     *
     * @note: positiveness of the result
     * given u = Av/‖Av‖ and v' = Aᵀu/‖Aᵀu‖ = Aᵀ(Av/‖Av‖)/‖Aᵀ(Av/‖Av‖)‖ = AᵀAv/‖AᵀAv‖
     * then uᵀAv' = (Av/‖Av‖)ᵀ A (AᵀAv/‖AᵀAv‖) = (AᵀAv)ᵀ(AᵀAv)/(‖Av‖⋅‖AᵀAv‖)
     *            = ‖AᵀAv‖²/(‖Av‖⋅‖AᵀAv‖) = ‖AᵀAv‖/‖Av‖ ≥ 0
     * likewise, if we start the iteration with v = Aᵀu/‖Aᵀu‖, then vᵀAᵀu' = ‖AAᵀu‖/‖Aᵀu‖ ≥ 0
     *
     * These actually suggest a different iteration scheme:
     * u <- Av
     * v <- Aᵀu
     * σ ← ‖v‖/‖u‖
     * u <- u/‖u‖
     * v <- v/‖v‖
     * The disadvantage here is that if σ is that ‖v‖ = 𝓞(σ²).
     *
     **/

    static Tensor forward(
        AutogradContext *ctx,
        const Tensor &A_in,
        const optional<Tensor> &u0,
        const optional<Tensor> &v0,
        const optional<int64_t> &maxiter,
        const double atol = 1e-6,
        const double rtol = 1e-6
    ) {
        /** @brief Forward pass.
         *
         * @param ctx: context object
         * @param A: m x n matrix
         * @param u0: initial guess for left singular vector
         * @param v0: initial guess for right singular vector
         * @param maxiter: maximum number of iterations
         * @param atol: absolute tolerance
         * @param rtol: relative tolerance
         * @returns sigma: singular value
         */
        // TODO: Test Anderson Acceleration

        // Sec: Option parsing
        constexpr int64_t DEFAULT_MAXITER = 256;
        const int64_t MAXITER = maxiter ? maxiter.value() : DEFAULT_MAXITER;
        const int64_t M = A_in.size(0);
        const int64_t N = A_in.size(1);
        const auto OPTIONS = A_in.options();
        bool converged = false;
        const Tensor ATOL = torch::scalar_tensor(atol, OPTIONS);
        const Tensor RTOL = torch::scalar_tensor(rtol, OPTIONS);

        // Preconditioning: normalize A by its infinity norm
        const Tensor SCALE = A_in.abs().max();
        const Tensor A = A_in / SCALE;
        const Tensor A_t = A.transpose(-2, -1);

        // pre-allocate buffers
        Tensor grad_u = torch::empty({M}, OPTIONS);
        Tensor grad_v = torch::empty({N}, OPTIONS);
        Tensor gamma_u = torch::empty({}, OPTIONS);
        Tensor gamma_v = torch::empty({}, OPTIONS);
        Tensor sigma_u = torch::empty({}, OPTIONS);
        Tensor sigma_v = torch::empty({}, OPTIONS);

        Tensor u = u0 ? u0.value() : torch::randn({M}, OPTIONS);
        Tensor v = v0 ? v0.value() : torch::randn({N}, OPTIONS);
        at::div_out(u, u, u.norm());
        at::div_out(v, v, v.norm());
        Tensor sigma = torch::zeros({}, OPTIONS);

        // special case: if SCALE == 0, then A is the zero matrix,
        // and the spectral norm is 0. We can return early to avoid NaNs in the iteration.
        if (SCALE.item<double>() == 0) {
			ctx->save_for_backward({u, v});
			return sigma;
		}

        // Perform power-iteration for maxiter times or until convergence.
        // NOTE: performing at least 2 iterations before the first convergence check is crucial,
        //   since only after two iterations one can guarantee that ⟨u∣Av⟩ > 0 and ⟨v∣Aᵀu⟩ > 0
        for (int64_t i = 0; i<MAXITER; i++) {
			// NOTE: Perform multiple iterations per loop to increase performance.
			//  Checking convergence is expensive, since `.item<bool>()` requires sync with CPU.
			//   The compiler cannot do this optimization on it's own because it would change behavior.
            #pragma unroll
            for (auto j = 0; j<7; j++) {
                // update u
                at::mv_out(grad_u, A, v);               // gᵤ ← Av
                at::div_out(u, grad_u, grad_u.norm());  // u ← gᵤ/‖gᵤ‖
                // update v
                at::mv_out(grad_v, A_t, u);             // gᵥ ← Aᵀu
                at::div_out(v, grad_v, grad_v.norm());  // v ← gᵥ/‖gᵥ‖
            }
            // update u
            at::mv_out(grad_u, A, v);                    // gᵤ ← Av
            at::dot_out(sigma_u, grad_u, u);             // σᵤ ← ⟨u∣gᵤ⟩
            at::norm_out(gamma_u, grad_u - sigma_u * u);  // γᵤ ← ‖gᵤ - σᵤu‖
            at::div_out(u, grad_u, grad_u.norm());       // u ← gᵤ/‖gᵤ‖
            // update v
            at::mv_out(grad_v, A_t, u);                  // gᵥ ← Aᵀu
            at::dot_out(sigma_v, grad_v, v);             // σᵥ ← ⟨v∣gᵥ⟩
            at::norm_out(gamma_v, grad_v - sigma_v * v); // γᵥ ← ‖gᵥ - σᵥv‖
            at::div_out(v, grad_v, grad_v.norm());       // v ← gᵥ/‖gᵥ‖

            // check convergence
            if ((converged = (
                  (gamma_u < (ATOL + RTOL * sigma_u))
                & (gamma_v < (ATOL + RTOL * sigma_v))
                ).item<bool>())
            ) {break;}
        }

        // Emit warning if no convergence within maxiter iterations.
        if (!converged) {
            TORCH_WARN("No convergence in ", MAXITER, " iterations for input of shape ", A.sizes());
        }

        // compute pre-conditioned sigma
        sigma = SCALE * A.mv(v).dot(u);

        // check for NaNs, infinities and non-positive values
        if ((~torch::isfinite(sigma) | (sigma <= 0)).item<bool>()) {
            throw std::runtime_error(at::str(
                "Computation resulted in invalid singular value σ=", sigma,
                " for input of shape ", A.sizes(), ". ",
                "Try increasing the number of iterations or the tolerance. ",
                "Currently maxiter=", MAXITER , ", atol=" , atol,  ", rtol=" , rtol , "."
            ));
        }

        // store pre-conditioned tensors for backward
        ctx->save_for_backward({u, v});

        return sigma;
    }

    static variable_list backward(
        const AutogradContext *ctx,
        const variable_list &grad_output
    ) {
        /** @brief Backward Pass.
         *
         * @param ctx: context object
         * @param grad_output: outer gradients
         * @returns g: gradient with respect to inputs
         *
         * Analytically, the VJP is ξ ↦ ξ⋅uvᵀ
         *
         */
        const auto saved = ctx->get_saved_variables();
        const Tensor &u = saved[0];
        const Tensor &v = saved[1];
        const Tensor g_sigma = grad_output[0] * outer(u, v);
        return { g_sigma, Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
    }
};


Tensor spectral_norm_meta(
    const Tensor &A,
    const optional<Tensor> &u0,
    const optional<Tensor> &v0,
    const optional<int64_t> &maxiter,
    const double atol,
    const double rtol
) {
    /**
     * Meta function for spectral norm.
     * This function is used to check the validity of the inputs and to infer the output shape and dtype.
     */
    TORCH_CHECK(A.dim() == 2, "Input must be a 2D matrix.");
    TORCH_CHECK(A.is_floating_point(), "Input must be a floating point tensor.");
    const auto M = A.size(0);
    const auto N = A.size(1);
    if (u0.has_value()) {
        TORCH_CHECK(u0.value().sizes() == torch::IntArrayRef({M}), "u0 must have shape (M,).");
        TORCH_CHECK(u0.value().dtype() == A.dtype(), "u0 must have the same dtype as A.");
    }
    if (v0.has_value()) {
        TORCH_CHECK(v0.value().sizes() == torch::IntArrayRef({N}), "v0 must have shape (N,).");
        TORCH_CHECK(v0.value().dtype() == A.dtype(), "v0 must have the same dtype as A.");
    }
    if (maxiter.has_value()) {
        TORCH_CHECK(maxiter.value() > 0, "maxiter must be a positive integer.");
    }
    return torch::empty({}, A.options());
}


Tensor spectral_norm(
    const Tensor &A,
    const optional<Tensor> &u0,
    const optional<Tensor> &v0,
    const optional<int64_t> &maxiter,
    const double atol,
    const double rtol
) {
    /**
     * Wrap the struct into function.
     */
    return SpectralNorm::apply(A, u0, v0, maxiter, atol, rtol);
}

}  // namespace linodenet_special


TORCH_LIBRARY_FRAGMENT(linodenet_special, m) {
    m.def(
        "spectral_norm("
            "Tensor A,"
            "Tensor? u0=None,"
            "Tensor? v0=None,"
            "int? maxiter=None,"
            "float atol=1e-6,"
            "float rtol=1e-6"
        ") -> Tensor"
    );
}

TORCH_LIBRARY_IMPL(linodenet_special, Autograd, m) {
    m.impl("spectral_norm", &linodenet_special::spectral_norm);
}

TORCH_LIBRARY_IMPL(linodenet_special, Meta, m) {
    m.impl("spectral_norm", &linodenet_special::spectral_norm_meta);
}
