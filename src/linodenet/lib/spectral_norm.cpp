#include <ATen/ATen.h>
#include <c10/util/irange.h>
#include <torch/script.h>
#include <torch/linalg.h>
//#include <cstddef>
//#include <string>


//import someLib as sl      ⟶  namespace sl = someLib;
//from someLib import func  ⟶  using someLib::func;
//from someLib import *     ⟶  using namespace someLib;

using c10::optional;
using torch::Tensor;
using torch::outer;
using torch::dot;
using torch::linalg::solve;

struct SpectralNorm: public torch::autograd::Function<SpectralNorm> {
    /** test
     * Formalizing as a optimization problem:
     * By Eckard-Young Theorem: min_{u,v} ‖A - σuvᵀ‖_F^2 s.t. ‖u‖₂ = ‖v‖₂ = 1
     * Equivalently: max_{u,v} ⟨A∣uv^⊤⟩ s.t. ‖u‖₂ = ‖v‖₂ = 1
     *
     * This is a non-convex QCQP, in standard form:
     * max_{(u,v)}  ½ [u, v]ᵀ [[0, A], [Aᵀ, 0]] [u, v]
     * s.t. [u, v]ᵀ [[𝕀ₘ, 0], [0, 0]] [u, v] - 1 =0
     * and  [u, v]ᵀ [[0, 0], [0, 𝕀ₙ]] [u, v] - 1 =0
     *
     * Related:
     * - https://math.stackexchange.com/questions/4658991
     * - https://math.stackexchange.com/questions/4697688
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
     */

    static Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const Tensor& A,
        optional<Tensor> u0,
        optional<Tensor> v0,
        optional<int64_t> maxiter,
        double atol = 1e-8,
        double rtol = 1e-5
    ) {
        /**
         * INPUTS:
         * @param ctx: context object
         * @param A: m x n matrix
         * @param u0: initial guess for left singular vector
         * @param v0: initial guess for right singular vector
         * @param maxiter: maximum number of iterations
         * @param atol: absolute tolerance
         * @param rtol: relative tolerance
         * @return
         * OUTPUTS:
         * sigma: singular value
         */
        // Initialize maxiter depending on the size of the matrix.
        const auto m = A.size(0);
        const auto n = A.size(1);
        const auto MAXITER = maxiter.has_value() ? maxiter.value() : 4*(m + n);
        bool converged = false;

        // Initialize u and v with random values if not given
        Tensor u = u0.has_value() ? u0.value() : torch::randn({m}, A.options());
        Tensor v = v0.has_value() ? v0.value() : torch::randn({n}, A.options());
        Tensor sigma = A.mv(v).dot(u);

        // Perform power-iteration for maxiter times or until convergence.
        for (const auto i : c10::irange(MAXITER)) {
            Tensor u_old = u;
            Tensor v_old = v;

            u = A.mv(v);
            sigma = dot(u, u_old);
            Tensor left_residual = (u - sigma * u_old).norm();
            u /= u.norm();
            // assert(sigma.item().toDouble() > 0);  // TODO: is it clear this never happens?!

            v = A.t().mv(u);
            sigma = dot(v, v_old);
            Tensor right_residual = (v - sigma * v_old).norm();
            v /= v.norm();
            // assert(sigma.item().toDouble() > 0);

            Tensor tol = atol + rtol * sigma;
            converged = (left_residual < tol).item<bool>() && (right_residual < tol).item<bool>();
            if (converged) {
                break;
            }
        }
        // Emit warning if no convergence within maxiter iterations.
        if (!converged) {
            TORCH_WARN("Spectral norm estimation did not converge in ", MAXITER, " iterations.")
        }
        assert(sigma.item<double>() > 0);
        // After convergence, we have: Av = σu, Aᵀu = σv. Thus σ = uᵀAv.
        ctx->save_for_backward({u, v});
        return sigma;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_output
    ) {
        /** Backward Pass.
         * Analytically, the VJP is ξ ↦ ξ⋅uvᵀ
         *
         * @param ctx: context object
         * @param grad_output: outer gradients
         * @return gradient with respect to inputs
         */
        auto saved = ctx->get_saved_variables();
        auto u = saved[0];
        auto v = saved[1];
        auto outer_grad = grad_output[0];
        auto g_sigma = outer_grad * outer(u, v);
        return { g_sigma, Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
    }
};

Tensor spectral_norm(
    const Tensor& A,
    const optional<Tensor>& u0,
    const optional<Tensor>& v0,
    optional<int64_t> maxiter,
    double atol = 1e-8,
    double rtol = 1e-5
) {
    /**
     * Wrap the struct into function.
     */
    return SpectralNorm::apply(A, u0, v0, maxiter, atol, rtol);
}

TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.def(
        "spectral_norm(Tensor A, Tensor? u0=None, Tensor? v0=None, int? maxiter=None, float atol=1e-8, float rtol=1e-5) -> Tensor",
        spectral_norm
    );
}
