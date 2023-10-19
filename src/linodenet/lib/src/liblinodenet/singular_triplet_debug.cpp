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


struct SingularTripletDebug : public Function<SingularTripletDebug> {
    /** @brief Compute the singular triplet of a matrix.
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
     * σ <- ‖v‖/‖u‖
     * u <- u/‖u‖
     * v <- v/‖v‖
     * The disadvantage here is that if σ is that ‖v‖ = 𝓞(σ²).
     *
     * @note (Stopping criterion):
     *     The normalized gradient stopping criterion is ‖∇L‖ ≤ α + β⋅‖L‖,
     *     which takes into account the magnitude of the gradient and the magnitude of the function value.
     *     When working with IEEE 754 floating point numbers, we generally need to use a relative tolerance,
     *     since the absolute tolerance depends on the magnitude of the function value.
     *
     *     Thus, the condition for convergence is
     *
     *     (I)     ‖Avₖ-σₖuₖ‖ ≤ α + β⋅|σₖ| and ‖Aᵀuₖ-σₖvₖ‖ ≤ α + β⋅|σₖ|
     *
     *     Assuming that σₖ≥0 this simplifies towards
     *
     *     (II)    ‖Avₖ-σₖuₖ‖ ≤ α + β⋅σₖ and ‖Aᵀuₖ-σₖvₖ‖ ≤ α + β⋅σₖ
     *
     *     Moreover, we can substitute ũₖ﹢₁=Avₖ and ṽₖ﹢₁=Aᵀuₖ to get
     *
     *     (III)   ‖ũₖ﹢₁ - σₖuₖ‖ ≤ α + β⋅σₖ and ‖ṽₖ﹢₁ - σₖvₖ‖ ≤ α + β⋅σₖ
     *
     *     secondly, we note that σₖ = ‖ũₖ‖ or σₖ = ‖ṽₖ‖, and uₖ = ũₖ/‖ũₖ‖ and vₖ = ṽₖ/‖ṽₖ‖,
     *     i.e. σₖuₖ = ũₖ and σₖvₖ = ṽₖ. Ergo (III) simplifies to
     *
     *     (IV)   ‖ũₖ﹢₁ - ũₖ‖ ≤ α + β ‖ũₖ‖ and ‖ṽₖ﹢₁ - ṽₖ‖ ≤ α + β ‖ṽₖ‖
     *
     *     The disadvantage of (IV) is the additional memory requirement for storing ũₖ and ṽₖ.
     *     Instead, we could also write:
     *
     *     (V)    ‖σₖ₊₁uₖ₊₁ - σₖuₖ‖ ≤ α + β⋅σₖ and ‖σₖ₊₁vₖ₊₁ - σₖvₖ‖ ≤ α + β⋅σₖ
     *
     *     which raises the question whether normalized or non-normalized vectors should be
     *     memorized between iterations. We allow the user to specify an initial guess,
     *     which begs the question whether the initial guess should be assumed to be normalized or not.
     *     In particular, if no initial guess is given, then the random initialization must be normalized.
     *
     * @note (PL-condition):
     *      The PL-condition is ½‖∇²f(x)‖² ≥ C (f(x) - f⁎),
     *      i.e. the gradient grows as a quadratic function of the distance to the optimum.
     *
     *      Combining with the assumption that ∇f is lipschitz continuous with constant L,
     *      one can prove that the gradient descent method converges with rate 𝓞(1/k),
     *      even when f is non-convex.
     **/

    static std::vector<Tensor> forward(
        AutogradContext *ctx,
        const Tensor &A,
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
         * @returns sigma, u, v: singular value, left singular vector, right singular vector
         */
        // Initialize maxiter depending on the size of the matrix.
        const auto A_t = A.t();
        const auto m = A.size(0);
        const auto n = A.size(1);
        const int64_t MAXITER = maxiter ? maxiter.value() : std::max<int64_t>(128, 2*(m + n));
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

        // Compute initial sigma. NOTE: absolute value ensures non-negativity.
        Tensor sigma = torch::abs(A.mv(v).dot(u));

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
        sigma = A.mv(v).dot(u);

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


std::tuple<Tensor, Tensor, Tensor> singular_triplet_debug(
    const Tensor  &A,
    const optional<Tensor> &u0,
    const optional<Tensor> &v0,
    optional<int64_t> maxiter,
    double atol = 1e-6,
    double rtol = 1e-6
) {
    /**
     * Wrap the struct into function.
     */
    auto output = SingularTripletDebug::apply(A, u0, v0, maxiter, atol, rtol);
    // assert(output.size() == 3);
    return std::make_tuple(output[0], output[1], output[2]);
}


TORCH_LIBRARY_FRAGMENT(liblinodenet, m) {
    m.def(
        "singular_triplet_debug("
            "Tensor A,"
            "Tensor? u0=None,"
            "Tensor? v0=None,"
            "int? maxiter=None,"
            "float atol=1e-6,"
            "float rtol=1e-6"
        ") -> (Tensor, Tensor, Tensor)",
        singular_triplet_debug
    );
}
