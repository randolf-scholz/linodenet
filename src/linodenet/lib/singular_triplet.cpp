#include <ATen/ATen.h>
#include <c10/util/irange.h>
#include <torch/script.h>
#include <torch/linalg.h>
#include <cstddef>
#include <string>
#include <vector>

//import someLib as sl      ⟶  namespace sl = someLib;
//from someLib import func  ⟶  using someLib::func;
//from someLib import *     ⟶  using namespace someLib;

using torch::Tensor;
using c10::optional;
using torch::linalg::solve;
using torch::outer;
using torch::dot;

struct SingularTriplet: public torch::autograd::Function<SingularTriplet> {
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

    static std::vector<Tensor> forward(
        torch::autograd::AutogradContext *ctx,
        Tensor A,
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
        const int m = A.size(0);
        const int n = A.size(1);
        int64_t MAXITER = maxiter.has_value() ? maxiter.value() : m + n;

        // Initialize u and v with random values if not given
        Tensor u = u0.has_value() ? u0.value() : torch::randn({m}, A.options());
        Tensor v = v0.has_value() ? v0.value() : torch::randn({n}, A.options());
        Tensor sigma = A.mv(v).dot(u);
        bool converged = false;

        // Perform power-iteration for maxiter times or until convergence.
        for (const auto i : c10::irange(MAXITER)) {
            Tensor u_old = u;
            Tensor v_old = v;

            u = A.mv(v);
            sigma = dot(u,u_old);
            Tensor left_residual = (u - sigma * u_old).norm();
            u /= u.norm();
            assert(sigma.item().toDouble() > 0);  // TODO: is it clear this never happens?!

            v = A.t().mv(u);
            sigma = dot(v, v_old);
            Tensor right_residual = (v - sigma * v_old).norm();
            v /= v.norm();
            assert(sigma.item().toDouble() > 0);

            Tensor tol = atol + rtol * sigma;
            converged = (left_residual < tol).item().toBool() && (right_residual < tol).item().toBool();
            if (converged) {break;}
        }
        // Emit warning if no convergence within maxiter iterations.
        if (!converged) {
            TORCH_WARN("Spectral norm estimation did not converge in ", MAXITER, " iterations.");
        }
        // After convergence, we have: Av = σu, Aᵀu = σv. Thus σ = uᵀAv.
        ctx->save_for_backward({A, sigma, u, v});
        auto result = {sigma, u, v};
        return result;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_output
    ) {
        /** Backward Pass.
         * INPUTS:
         * @param ctx: context object
         * @param grad_output: outer gradients
         * @return gradient with respect to inputs
         *
         * Analytically, the VJPs are
         * ξᵀ(∂σ/∂A) = ξ⋅uvᵀ
         * Φᵀ(∂u/∂A) = (𝕀ₘ-uuᵀ)Φ'vᵀ
         * Ψᵀ(∂v/∂A) = uΨ'(𝕀ₙ-vvᵀ)
         *
         * Here, Φ' and Ψ' are given as the solutions to the linar system
         * [σ𝕀ₘ, -Aᵀ]  [Φ'] = [Φ]
         * [-Aᵀ, σ𝕀ₙ]  [Ψ'] = [Ψ]
         *
         * We can use the formula for the 2x2 block inverse to see that we can solve 4 smaller systems instead.
         *  [𝕀ₘ - BBᵀ]x = Φ  [𝕀ₙ - BᵀB]y = BΨ
         *  [𝕀ₙ - BᵀB]w = BᵀΦ  [𝕀ₘ - BBᵀ]z = Ψ
         *
         */
        auto saved = ctx->get_saved_variables();
        auto A = saved[0];
        auto sigma = saved[1];
        auto u = saved[2];
        auto v = saved[3];
        auto xi = grad_output[0];
        auto phi = grad_output[1];
        auto psi = grad_output[2];

        const int m = A.size(0);
        const int n = A.size(1);
        A /= sigma;

        // Compute the 2x2 block inverses
        Tensor P = torch::eye(m) - A.mm(A.t());
        Tensor Q = torch::eye(n) - A.t().mm(A);

        Tensor x = solve(P, phi, true);
        Tensor y = solve(P, A.dot(psi), true);
        Tensor w = solve(Q, A.t().dot(phi), true);
        Tensor z = solve(Q, psi, true);

        phi = (x+y)/sigma;
        psi = (w+z)/sigma;
        Tensor g_sigma = xi * outer(u, v);
        Tensor g_u = outer(phi - dot(u, phi)*u, v);
        Tensor g_v = outer(u, psi - dot(v, psi)*v);
        torch::autograd::variable_list output = {g_sigma, g_u, g_v};
        return output;
    }
};

std::vector<Tensor> singular_triplet(
        Tensor A,
        optional<Tensor> u0,
        optional<Tensor> v0,
        optional<int64_t> maxiter,
        double atol = 1e-8,
        double rtol = 1e-5
) {
    /**
     * Wrap the struct into function.
     */
    return SingularTriplet::apply(A, u0, v0, maxiter, atol, rtol);
}

TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.def(
        "singular_triplet(Tensor A, Tensor? u0=None, Tensor? v0=None, int? maxiter=None, float atol=1e-8, float rtol=1e-5) -> tuple[Tensor, Tensor, Tensor]",
        singular_triplet
    );
}
