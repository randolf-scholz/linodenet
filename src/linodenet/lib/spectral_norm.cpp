#include <ATen/ATen.h>
#include <c10/util/irange.h>
#include <torch/script.h>
#include <torch/linalg.h>
#include <cstddef>
#include <string>


//import someLib as sl      ⟶  namespace sl = someLib;
//from someLib import func  ⟶  using someLib::func;
//from someLib import *     ⟶  using namespace someLib;

using torch::linalg::vector_norm;
using torch::Tensor;
using c10::optional;

struct SpectralNorm: public torch::autograd::Function<SpectralNorm> {
    // Formalizing as a optimization problem: min ‖A - σuvᵀ‖_F^2 s.t. ‖u‖₂ = ‖v‖₂ = 1
    // This we can actually simplify toward max_{u,v} σ = uᵀAv s.t. ‖u‖² = ‖v‖² = 1

    // Lagrangian: L(u,v,λ,μ) = uᵀAv - λ(uᵀu - 1) - μ(vᵀv - 1)
    // KKT conditions: ∇L = 0 ⟺ A v - 2λu = 0 ⟺ [-2λ𝕀ₘ, A    ] [u] = [0]
    //                          Aᵀu - 2μv = 0   [Aᵀ   , -2μ𝕀ₙ] [v] = [0]
    //
    // Second order conditions:  sᵀ∇²Ls ≥ 0 uf ∇hᵀs = 0
    // ∇hᵀ = [2uᵀ, 2vᵀ]
    // ∇²L =  [-2λ𝕀ₘ, A    ]
    //        [Aᵀ   , -2μ𝕀ₙ]
    // NOTE: the gradient is linear, and the problem is a quadratic optimization problem!
    // in particular, the problem can be solved by a single Newton step!
    //

    // Equality constrained optimization problem:

    // max_{(u,v)}  ½ [u, v]ᵀ [[0, A], [Aᵀ, 0]] [u, v] s.t. uᵀu = 1, vᵀv = 1




    // The first order convergence criterion is ‖Av-σu‖₂ = 0 and ‖Aᵀu-σv‖₂ = 0
    // Plugging in the iteration, we get ‖u' - σũ‖ = 0 and ‖v' - σṽ‖ = 0 (tilde indicates normalized vector)
    // secondly we can estimate σ in each iteration via one of the 3 formulas
    // (1) σ = uᵀAv  (2) σᵤ = ũᵀu'  (3) σᵥ = ṽᵀv'
    // Plugging these into the equations we get
    // ‖u' -  u'ᵀ ũᵀũ‖


    // Error estimate: Note that
    // ‖Av - σu‖ = ‖σ̃ũ - σu‖ = ‖σ̃ũ - σũ + σũ -σu‖ ≤ ‖σ̃ũ - σũ‖ + ‖σũ -σu‖ = (σ̃ - σ) + σ‖ũ - u‖



    static Tensor forward(
        torch::autograd::AutogradContext *ctx,
        Tensor A,
        optional<Tensor> u0,
        optional<Tensor> v0,
        optional<int64_t> maxiter,
        double atol = 1e-8,
        double rtol = 1e-5
    ) {
        // MEMORY USAGE: 2 * m + 2 * n
        // MVs per iteration: 2

        // initialize maxiter depending on the size of the matrix
        const int m = A.size(0);
        const int n = A.size(1);
        int64_t MAXITER = maxiter.has_value() ? maxiter.value() : m + n;

        // Initialize u and v with random values if not given
        Tensor u = u0.has_value() ? u0.value() : torch::randn({m}, A.options());
        Tensor v = v0.has_value() ? v0.value() : torch::randn({n}, A.options());
        Tensor sigma = A.mv(v).dot(u);
        bool converged = false;

        // perform power-iteration for maxiter times or until convergence.
        for (const auto i : c10::irange(MAXITER)) {
            Tensor u_old = u;
            Tensor v_old = v;

            u = A.mv(v);
            sigma = u.dot(u_old);
            Tensor left_residual = (u - sigma * u_old).norm();
            u /= u.norm();
            assert(sigma.item().toDouble() > 0);

            v = A.t().mv(u);
            sigma = v.dot(v_old);
            Tensor right_residual = (v - sigma * v_old).norm();
            v /= v.norm();
            assert(sigma.item().toDouble() > 0);

            Tensor tol = atol + rtol * sigma;
            converged = (left_residual < tol).item().toBool() && (right_residual < tol).item().toBool();
            if (converged) {break;}
        }
        // emit warning if no convergence within maxiter iterations
        if (!converged) {
            TORCH_WARN("Spectral norm estimation did not converge in ", MAXITER, " iterations.");
        }
        // after convergence, we have: Av = σu, Aᵀu = σv. Thus σ = uᵀAv
        ctx->save_for_backward({u, v});
        return sigma;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_output
    ) {
        auto saved = ctx->get_saved_variables();
        auto u = saved[0];
        auto v = saved[1];
        auto outer_grad = grad_output[0];
        torch::autograd::variable_list output = {
            outer_grad * u.outer(v),
//            torch::einsum("..., i, j -> ...ij", outer_grad, u, v);
        };
        return output;
    }
};

Tensor spectral_norm(
        Tensor A,
        optional<Tensor> u0,
        optional<Tensor> v0,
        optional<int64_t> maxiter,
        double atol = 1e-8,
        double rtol = 1e-5
) {
    return SpectralNorm::apply(A, u0, v0, maxiter, atol, rtol);
}

TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.def("spectral_norm(Tensor A, Tensor? u0=None, Tensor? v0=None, int? maxiter=None, float atol=1e-8, float rtol=1e-5) -> Tensor", spectral_norm);
}
