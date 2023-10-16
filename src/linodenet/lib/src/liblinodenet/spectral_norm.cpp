// #include <ATen/ATen.h>
#include <torch/script.h>
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

/*
 * NOTE: discontinuity of singular values.
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
 * However, there should be "good" paths, that continuously deform the singular dyads.
 * Case in point: when A is the identity matrix, then **every** vector is a singular vector.
 *
 * */


struct SpectralNorm: public Function<SpectralNorm> {
    /** @brief Spectral norm of a matrix.
     *
     * Formalizing as a optimization problem:
     * By Eckard-Young Theorem: min_{u,v} ‖A - σuvᵀ‖_F^2 s.t. ‖u‖₂ = ‖v‖₂ = 1
     * Equivalently: max_{u,v} ⟨A∣uv^⊤⟩ s.t. ‖u‖₂ = ‖v‖₂ = 1
     *
     * This is a non-convex QCQP, in standard form:
     * max_{(u,v)}  ½ [u, v]ᵀ [[0, A], [Aᵀ, 0]] [u, v]
     * s.t. [u, v]ᵀ [[𝕀ₘ, 0], [0, 0]] [u, v] - 1 =0
     * and  [u, v]ᵀ [[0, 0], [0, 𝕀ₙ]] [u, v] - 1 =0
     *
     * @related https://math.stackexchange.com/questions/4658991
     * @related https://math.stackexchange.com/questions/4697688
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
        optional<int64_t> maxiter,
        double atol = 1e-6,
        double rtol = 1e-6
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

//        // if no initial guess is given, use builtin svd.
//        if (!u0 || !v0) {
//            auto [U, S, Vh] = torch::linalg_svd(A_in);
//            Tensor u = U.index({0});
//            Tensor v = Vh.index({"...", 0});
//            Tensor sigma = S.index({0});
//            // store pre-conditioned tensors for backward
//            ctx->save_for_backward({u, v});
//            return sigma;
//        }

        // Initialize maxiter depending on the size of the matrix.
        const auto M = A_in.size(0);
        const auto N = A_in.size(1);
        const auto OPTIONS = A_in.options();
        const int64_t MAXITER = maxiter ? maxiter.value() : (M + N + 64);

        // Initialize tolerance scalars
        const Tensor ATOL = torch::full({}, atol, OPTIONS);
        const Tensor RTOL = torch::full({}, rtol, OPTIONS);

        // Preconditioning: normalize A by its infinity norm
        const Tensor SCALE = A_in.abs().max();
        const auto A = A_in / SCALE;
        const auto A_t = A.t();  // precompute transpose (maybe skip for small MAXITER?)

        // Initialize convergence flag
        bool converged = false;

        // Initialize u and v with random values if not given
        Tensor u = u0 ? u0.value() : torch::randn({M}, OPTIONS);
        Tensor v = v0 ? v0.value() : torch::randn({N}, OPTIONS);

        // initialize vectors for power iteration
        v /= v.norm();
        u /= u.norm();

        // pre-allocate buffers
        Tensor grad_u = torch::empty_like(u);
        Tensor grad_v = torch::empty_like(v);

        // scalars
        Tensor gamma_u = torch::empty({}, OPTIONS);
        Tensor gamma_v = torch::empty({}, OPTIONS);
        Tensor sigma_u = torch::empty({}, OPTIONS);
        Tensor sigma_v = torch::empty({}, OPTIONS);

        // Perform power-iteration for maxiter times or until convergence.
        // NOTE: Perform 2 iterations per loop to increase performance.
        //  Checking convergence is expensive, since `.item<bool>()` requires sync with CPU.
        //   The compiler cannot do this optimization on it's own because it would change behavior.
        // NOTE: performing at least 2 iterations before the first convergence check is crucial,
        //   since only after two iterations one can guarantee that ⟨u∣Av⟩ > 0 and ⟨v∣Aᵀu⟩ > 0
        for (auto i = 0; i<MAXITER; i++) {
            #pragma unroll  // we test convergence only every 8th iteration.
            for (auto j = 0; j<7; j++) {
                // update u
                u = A.mv(v);
                u /= u.norm();
                // update v
                v = A_t.mv(u);
                v /= v.norm();
            }
            // update u
            grad_u = A.mv(v);
            sigma_u = grad_u.dot(u);
            gamma_u = (grad_u - sigma_u * u).norm();
            u = grad_u / grad_u.norm();
            // update v
            grad_v = A_t.mv(u);
            sigma_v = grad_v.dot(v);
            gamma_v = (grad_v - sigma_v * v).norm();
            v = grad_v / grad_v.norm();

            // check convergence
            // NOTE:(1/√2)(‖u‖+‖v‖) ≤ ‖(u,v)‖ ≤ ‖u‖+‖v‖ (via Jensen-Inequality)
            if ((converged = (  // compare against geometric mean
                torch::max(gamma_u, gamma_v) < (ATOL + RTOL*torch::min(sigma_u, sigma_v))
            ).item<bool>())) {break;}
        }

        // Emit warning if no convergence within maxiter iterations.
        if (!converged) {
            TORCH_WARN("No convergence in ", MAXITER, " iterations for input of shape ", A.sizes())
        }

        // compute pre-conditioned sigma
        const Tensor sigma = A.mv(v).dot(u);

        // store pre-conditioned tensors for backward
        ctx->save_for_backward({u, v});

        // reverse pre-conditioning
        const Tensor sigma_out = sigma * SCALE;

        // check for NaNs, infinities and non-positive values
        if ((~torch::isfinite(sigma_out) | (sigma_out <= 0)).item<bool>()) [[unlikely]] {
            throw std::runtime_error(at::str(
                "Computation resulted in invalid singular value σ=", sigma_out,
                " for input of shape ", A.sizes(), ". ",
                "Try increasing the number of iterations or the tolerance. ",
                "Currently maxiter=", MAXITER , ", atol=" , atol,  ", rtol=" , rtol , "."
            ));
        }
        return sigma_out;
    }

    static variable_list backward(
        AutogradContext *ctx,
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
        auto saved = ctx->get_saved_variables();
        auto u = saved[0];
        auto v = saved[1];
        auto g_sigma = grad_output[0] * outer(u, v);
        return { g_sigma, Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
    }
};


static inline Tensor spectral_norm(
    const Tensor &A,
    const optional<Tensor> &u0,
    const optional<Tensor> &v0,
    optional<int64_t> maxiter,
    double atol = 1e-6,
    double rtol = 1e-6
) {
    /**
     * Wrap the struct into function.
     */
    return SpectralNorm::apply(A, u0, v0, maxiter, atol, rtol);
}


TORCH_LIBRARY_FRAGMENT(liblinodenet, m) {
    m.def(
        "spectral_norm("
            "Tensor A,"
            "Tensor? u0=None,"
            "Tensor? v0=None,"
            "int? maxiter=None,"
            "float atol=1e-6,"
            "float rtol=1e-6"
        ") -> Tensor",
        spectral_norm
    );
}
