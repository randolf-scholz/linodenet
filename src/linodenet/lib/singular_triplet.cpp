// #include <ATen/ATen.h>
#include <torch/script.h>
#include <torch/linalg.h>
#include <vector>
// #include <string>

// import someLib as sl      ⟶  namespace sl = someLib;
// from someLib import func  ⟶  using someLib::func;
// from someLib import *     ⟶  using namespace someLib;
using torch::optional;
using torch::nullopt;
using torch::Tensor;
using torch::Scalar;
using torch::cat;
using torch::outer;
using torch::dot;
using torch::eye;
using torch::addmm;
using torch::linalg::solve;
using torch::linalg::lstsq;
using torch::autograd::variable_list;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::indexing::Slice;


struct SingularTriplet : public Function<SingularTriplet> {
    /** @brief Compute the singular triplet of a matrix.
     *
     * @details Formalizing as a optimization problem:
     * By Eckard-Young Theorem: min_{u,v} ‖A - σuvᵀ‖_F^2 s.t. ‖u‖₂ = ‖v‖₂ = 1
     * Equivalently: max_{u,v} ⟨A∣uv^⊤⟩ s.t. ‖u‖₂ = ‖v‖₂ = 1
     *
     * @details This is a non-convex QCQP, in standard form:
     * max_{(u,v)}  ½ [u, v]ᵀ [[0, A], [Aᵀ, 0]] [u, v]
     * s.t. [u, v]ᵀ [[𝕀ₘ, 0], [0, 0]] [u, v] - 1 =0
     * and  [u, v]ᵀ [[0, 0], [0, 𝕀ₙ]] [u, v] - 1 =0
     *
     * @related https://math.stackexchange.com/questions/4658991
     * @related https://math.stackexchange.com/questions/4697688
     *
     * Lagrangian: L(u,v,λ,μ) = uᵀAv - λ(uᵀu - 1) - μ(vᵀv - 1)
     *
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
     *     The standard stopping criterion is ‖ũ - σu‖ ≤ α + β⋅σ
     *     Plugging in the definition of ũ and σ, and dividing by ‖ũ‖ yields, using u'=  ũ/‖ũ‖
     *     ‖u'-⟨u∣u'⟩u‖ ≤ α/‖ũ‖ + β ⟨u∣u'⟩
     *     close to convergence, ⟨u∣u'⟩ ≈ 1, giving the stopping criterion
     *     ‖u'-u‖ ≤ α/‖ũ‖ + β
     *     Assuming ‖ũ‖>1, we can the first term. Squaring gives the final criterion:i
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

    static std::vector<Tensor> forward(
        AutogradContext *ctx,
        const Tensor &A,
        const optional<Tensor> &u0,
        const optional<Tensor> &v0,
        optional<int64_t> maxiter,
        double atol = 1e-8,
        double rtol = 1e-5
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
         * @returns sigma, u, v: singular value, left singular vector, right singular vector
         */
        // Initialize maxiter depending on the size of the matrix.
        const auto A_t = A.t();
        const auto m = A.size(0);
        const auto n = A.size(1);
        const int64_t MAXITER = maxiter ? maxiter.value() : std::max<int64_t>(100, 2*(m + n));
        const Tensor tol = torch::tensor(rtol * rtol);
        bool converged = false;

        // Initialize u and v with random values if not given
        Tensor u = u0 ? u0.value() : torch::randn({m}, A.options());
        Tensor v = v0 ? v0.value() : torch::randn({n}, A.options());

        // Initialize old values for convergence check
        // pre-allocate memory for residuals
        Tensor u_old = torch::empty_like(u);
        Tensor v_old = torch::empty_like(v);
        Tensor r_u = torch::empty_like(u);
        Tensor r_v = torch::empty_like(v);

        // Perform power-iteration for maxiter times or until convergence.
        for (auto i = 0; i<MAXITER; i++) {
            // NOTE: We apply two iterations per loop. This is a case of duff's device.
            // This means that we effectively only check the stopping criterion every 2 iterations.
            // This improves performance on GPU since .item() requires a synchronization with CPU.
            // The compiler cannot do this optimization on it's own because it would change behavior.

            // update u
            u = A.mv(v);
            u /= u.norm();

            // update v
            v = A_t.mv(u);
            v /= v.norm();

            // update u
            u_old = u;
            u = A.mv(v);
            u /= u.norm();

            // update v
            v_old = v;
            v = A_t.mv(u);
            v /= v.norm();

            // performance: do not test convergence after evey iteration
            r_u = u - u_old;
            r_v = v - v_old;

            // check convergence
            if ((converged = ((r_v.dot(r_v) < tol) & (r_u.dot(r_u) < tol)).item<bool>())) {
                // Tensor sigma = A.mv(v).dot(u);
                // std::cout << "Converged after " << i << " iterations. Sigma=" << sigma.item<double>() << std::endl;
                break;
            }
        }

        // Emit warning if no convergence within maxiter iterations.
        if (!converged) {
            TORCH_WARN(": no convergence in ", MAXITER, " iterations for input of shape ", A.sizes())
        }

        // compute final sigma
        Tensor sigma = A.mv(v).dot(u);

        // check for NaNs, infinities, and negative values
        const auto sigma_val = sigma.item<double>();
        if (!(std::isfinite(sigma_val) && sigma_val > 0)) {
            throw std::runtime_error(at::str(
                "Computation resulted in invalid singular value σ=", sigma_val, " for input of shape ", A.sizes(), ". ",
                "Try increasing the number of iterations or the tolerance. ",
                "Currently maxiter=", MAXITER , ", atol=" , atol,  ", rtol=" , rtol , "."
            ));
        }

        // After convergence, we have: Av = σu, Aᵀu = σv. Thus σ = uᵀAv.
        ctx->save_for_backward({A, sigma, u, v});

        return {sigma, u, v};
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
         * Analytically, the VJPs are
         * ξᵀ(∂σ/∂A) = ξ⋅uvᵀ
         * Φᵀ(∂u/∂A) = (𝕀ₘ-uuᵀ)Φ'vᵀ
         * Ψᵀ(∂v/∂A) = uΨ'(𝕀ₙ-vvᵀ)
         *
         * Here, Φ' and Ψ' are given as the solutions to the linear system
         * [σ𝕀ₘ, -Aᵀ]  [Φ'] = [Φ]
         * [-Aᵀ, σ𝕀ₙ]  [Ψ'] = [Ψ]
         *
         * We can use the formula for the 2x2 block inverse to see that we can solve 4 smaller systems instead.
         *  [𝕀ₘ - BBᵀ]x = Φ  [𝕀ₙ - BᵀB]y = BΨ
         *  [𝕀ₙ - BᵀB]w = BᵀΦ  [𝕀ₘ - BBᵀ]z = Ψ
         */
        const auto saved = ctx->get_saved_variables();
        const auto A = saved[0];
        const auto sigma = saved[1];
        const auto u = saved[2];
        const auto v = saved[3];
        const auto xi = grad_output[0];
        const auto phi = grad_output[1];
        const auto psi = grad_output[2];

        // Computing reference values via SVD
        // auto SVD = torch::linalg::svd(A, true, nullopt);
        // Tensor u = std::get<0>(SVD).index({torch::indexing::Slice(), 0});
        // Tensor s = std::get<1>(SVD).index({0});
        // Tensor v = std::get<2>(SVD).index({0, torch::indexing::Slice()});

        Tensor g_sigma = xi * outer(u, v);

        // exit early if grad_output is zero for both u and v.
        if ( !(phi.any() | psi.any()).item<bool>() ) {
            return {g_sigma, Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
        }

        // Consider the additional outer gradients for u and v.
        const auto m = A.size(0);
        const auto n = A.size(1);
        // augmented K matrix: (m+n+2) x (m+n)
        // [ σ𝕀ₘ | -A  | u | 0 ] ⋅ [p, q, μ, ν] = [ϕ]
        // [ -Aᵀ | σ𝕀ₙ | 0 | v ]                  [ψ]

        const auto options = A.options();

        // construct the K matrix
//        Tensor K = torch::zeros({m+n, m+n+2}, options);
//        K.index_put_({Slice(0, m+n), Slice(0, m+n)}, sigma*eye(m+n, options));
//        K.index_put_({Slice(0, m), Slice(m, m+n)}, -A);
//        K.index_put_({Slice(m, m+n), Slice(0, m)}, -A.t());
//        K.index_put_({Slice(0, m), m+n}, u);
//        K.index_put_({Slice(m, m+n), m+n+1}, v);

        Tensor c = torch::cat({phi, psi}, 0);

        Tensor zero_u = torch::zeros_like(u).unsqueeze(-1);
        Tensor zero_v = torch::zeros_like(v).unsqueeze(-1);

        Tensor K = cat({
            cat({sigma * eye(m, options), -A, u.unsqueeze(-1), zero_u}, 1),
            cat({-A.t(), sigma * eye(n, options), zero_v, v.unsqueeze(-1)}, 1)
        }, 0);

        // solve the underdetermined system
        Tensor x = std::get<0>(lstsq(K, c, nullopt, nullopt));
        Tensor p = x.slice(0, 0, m);
        Tensor q = x.slice(0, m, m + n);
        // Tensor mu = x.slice(0, m+n, m+n+1);
        // Tensor nu = x.slice(0, m+n+1, m+n+2);

        // compute the VJP
        Tensor g_u = outer(p - dot(u, p) * u, v);
        Tensor g_v = outer(u, q - dot(v, q) * v);
        return { g_sigma + g_u + g_v, Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
    }
};

/**
 * Solving 2 m×m and 2 n×n systems instead.
 * torch::Scalar sigma2 = (sigma * sigma).item();
 * Tensor P = addmm(eye(m, A.options()), A, A.t(), sigma2, -1.0);  // σ²𝕀ₘ - AAᵀ
 * Tensor Q = addmm(eye(n, A.options()), A.t(), A, sigma2, -1.0);  // σ²𝕀ₙ - AᵀA
 *
 * Tensor x = std::get<0>(lstsq(P, sigma*phi, nullopt, nullopt));
 * Tensor w = std::get<0>(lstsq(Q, A.t().mv(phi), nullopt, nullopt));
 * Tensor y = std::get<0>(lstsq(P, A.mv(psi), nullopt, nullopt));
 * Tensor z = std::get<0>(lstsq(Q, sigma*psi, nullopt, nullopt));
 * Tensor p = x + y;
 * Tensor q = w + z;

 * Tensor x, y, z, w;
 * if (phi_nonzero) {
 *     x = std::get<0>(lstsq(P, sigma*phi, nullopt, nullopt));
 *     w = std::get<0>(lstsq(Q, A.t().mv(phi), nullopt, nullopt));
 * } else {
 *     x = torch::zeros_like(phi);
 *     w = torch::zeros_like(psi);
 * }
 * if (psi_nonzero) {
 *     y = std::get<0>(lstsq(P, A.mv(psi), nullopt, nullopt));
 *     z = std::get<0>(lstsq(Q, sigma*psi, nullopt, nullopt));
 * } else {
 *     y = torch::zeros_like(phi);
 *     z = torch::zeros_like(psi);
 * }
 */


std::tuple<Tensor, Tensor, Tensor> singular_triplet(
    const Tensor  &A,
    const optional<Tensor> &u0,
    const optional<Tensor> &v0,
    optional<int64_t> maxiter,
    double atol = 1e-8,
    double rtol = 1e-5
) {
    /**
     * Wrap the struct into function.
     */
    auto output = SingularTriplet::apply(A, u0, v0, maxiter, atol, rtol);
    // assert(output.size() == 3);
    return std::make_tuple(output[0], output[1], output[2]);
}


TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.def(
        "singular_triplet("
            "Tensor A,"
            "Tensor? u0=None,"
            "Tensor? v0=None,"
            "int? maxiter=None,"
            "float atol=1e-8,"
            "float rtol=1e-5"
        ") -> (Tensor, Tensor, Tensor)",
        singular_triplet
    );
}
