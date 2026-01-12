// #include <ATen/ATen.h>
#include <torch/torch.h>
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
using torch::linalg_lstsq;
using torch::autograd::variable_list;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::indexing::Slice;


struct SingularTriplet : public Function<SingularTriplet> {
    /** @brief Compute the singular triplet of a matrix.
     *
     * @details Formalizing as a optimization problem:
     * By Eckard-Young Theorem: min_{u,v} ‖A - σuvᵀ‖² s.t. ‖u‖₂ = ‖v‖₂ = 1
     * Equivalently: max_{u,v} ⟨A∣uvᵀ⟩ s.t. ‖u‖₂ = ‖v‖₂ = 1
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
     *     The normalized gradient stopping criterion is ‖∇L‖ ≤ α + β⋅‖L‖,
     *     which takes into account the magnitude of the gradient and the magnitude of the function value.
     *     In our case ‖L‖ = |vᵀAu| is the estimated singular value
     *     and ‖∇L‖² = ‖(Av-2λu, Aᵀu-2μv)‖² = ‖Av-2λu‖² + ‖Aᵀu-2μv‖² = ‖Av-σu‖² + ‖Aᵀu-σv‖²
     *
     *     Specifically, in the first residual σ=‖ũ‖ and in the second residual σ=‖ṽ‖.
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
     *     This suggests that one should normalize at the start of the for loop, as well as after
     *     the final iteration. In particular, this ensures that the prediction yields normalized
     *     vectors even when MAXITER is set to 0, and the loop is not executed at all.
     *
     * @note (PL-condition):
     *      The PL-condition is ½‖∇²f(x)‖² ≥ C (f(x) - f⁎),
     *      i.e. the gradient grows as a quadratic function of the distance to the optimum.
     *
     *      Combining with the assumption that ∇f is lipschitz continuous with constant L,
     *      one can prove that the gradient descent method converges with rate 𝓞(1/k),
     *      even when f is non-convex.
     *
     * @note Yet another stopping criterion:
     *      Fundamentally it is about finding uvᵀ, hence we should consider ‖ũṽᵀ - uvᵀ‖ ≤ α + β‖uvᵀ‖
     *      Assuming the vectors are normalized, and noting that ‖uvᵀ‖² = ‖u‖²‖v‖², we get
     *
     *      ‖ũṽᵀ - uvᵀ‖² = ‖ũ‖²‖ṽ‖² -2⟨ũ,u⟩⟨ṽ,v⟩ + ‖u‖²‖v‖² = 2(1-⟨ũ,u⟩⟨ṽ,v⟩)
     *
     *      Which simplifies the stopping criterion towards: ⟨ũ,u⟩⟨ṽ,v⟩ ≥ 2 - (α + β)²
     *      In particular, we could get away with only a single tolerance parameter ξ = α + β.
     *      Note that this has a failure mode: if ũ ≈ -u and ṽ ≈ -v, then the criterion is satisfied.
     *      But this can never happen in practice, since effectively ũ ∝ AAᵀu and ṽ ∝ AᵀAv.
     *      And both AAᵀ and AᵀA are positive semi-definite, hence ũ and ṽ can never be anti-parallel.
     *
     *      Adding σ into the equation:
     *
     *      ‖σ̃ũṽᵀ - σuvᵀ‖² = σ̃²‖ũ‖²‖ṽ‖² -2σ̃σ⟨ũ,u⟩⟨ṽ,v⟩ + σ²‖u‖²‖v‖²
     *                     = σ̃² + σ² - 2σ̃σ⟨ũ,u⟩⟨ṽ,v⟩
     *
     *      Our goal is to transform this into c²⋅(1+x), so that we can make use of the upper bound
     *      √(1+x) ≤ 1 + ½x and avoid the expensive square root.
     *
     *                     = (σ̃ - σ)² + 2σ̃σ(1 - ⟨ũ,u⟩⟨ṽ,v⟩)
     *                     = (σ̃ - σ)²(1 + 2⋅(σ̃σ/(σ̃ - σ)²)⋅(1 - ⟨ũ,u⟩⟨ṽ,v⟩))
     *                     = (σ̃ - σ)²(1 + x) where x=2(σ̃σ/(σ̃ - σ)²)(1 - ⟨ũ,u⟩⟨ṽ,v⟩)
     *
     *      Taking the square root and applying the upper bound yields
     *
     *      √(c²(1+x)) = |c|⋅√(1+x)
     *                 ≤ |c|⋅(1+½x) = |σ̃ - σ|⋅(1 + (σ̃σ/(σ̃ - σ)²)(1 - ⟨ũ,u⟩⟨ṽ,v⟩))
     *                              = |σ̃ - σ| + σ̃σ/|σ̃ - σ|(1 - ⟨ũ,u⟩⟨ṽ,v⟩)
     *                              = |σ̃ - σ| + (1 - ⟨ũ,u⟩⟨ṽ,v⟩)/|σ̃⁻¹-σ⁻¹|
     *
     *      In order for this to be small, we need |σ̃ - σ| to be small and ⟨ũ,u⟩⟨ṽ,v⟩ to be close to 1.
     *      Since the second term is scaled by |σ̃⁻¹-σ⁻¹|⁻¹, the residual in the vectors must be small.
     *      To avoid division by zero, we can use the harmonic mean instead of the arithmetic mean:
     *
     * @note: positiveness of the result
     * given u = Av/‖Av‖ and v' = Aᵀu/‖Aᵀu‖ = Aᵀ(Av/‖Av‖)/‖Aᵀ(Av/‖Av‖)‖ = AᵀAv/‖AᵀAv‖
     * then uᵀAv' = (Av/‖Av‖)ᵀ A (AᵀAv/‖AᵀAv‖) = (AᵀAv)ᵀ(AᵀAv)/(‖Av‖⋅‖AᵀAv‖)
     *            = ‖AᵀAv‖²/(‖Av‖⋅‖AᵀAv‖) = ‖AᵀAv‖/‖Av‖ = ‖Aᵀ(Av/‖Av‖)‖ = ‖Aᵀu‖ = ‖ṽ‖
     * likewise, if we start the iteration with v = Aᵀu/‖Aᵀu‖, then vᵀAᵀu' = ‖ũ‖ ≥ 0
     *
     * These actually suggest a different iteration scheme:
     * u <- Av
     * v <- Aᵀu
     * σ ← ‖v‖/‖u‖
     * u <- u/‖u‖
     * v <- v/‖v‖
     * This has the huge advantage that σ is guaranteed to be positive, whereas the other iteration scheme
     * can produce negative values for σ due to numerical errors.
     * The disadvantage here is that if σ is that ‖v‖ = 𝓞(σ²).
     *
     **/

    static std::vector<Tensor> forward(
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
         * @returns sigma, u, v: singular value, left singular vector, right singular vector
         */
        // TODO: Test Anderson Acceleration

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
        ctx->save_for_backward({A, sigma, u, v, SCALE});

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
        return {sigma_out, u, v};
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
        const auto SCALE = saved[4];
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
        const auto M = A.size(0);
        const auto N = A.size(1);
        const auto OPTIONS = A.options();
        // augmented K matrix: (m+n+2) x (m+n)
        // [ σ𝕀ₘ | -A  | u | 0 ] ⋅ [p, q, μ, ν] = [ϕ]
        // [ -Aᵀ | σ𝕀ₙ | 0 | v ]                  [ψ]


        // construct the K matrix
//        Tensor K = torch::zeros({m+n, m+n+2}, options);
//        K.index_put_({Slice(0, m+n), Slice(0, m+n)}, sigma*eye(m+n, options));
//        K.index_put_({Slice(0, m), Slice(m, m+n)}, -A);
//        K.index_put_({Slice(m, m+n), Slice(0, m)}, -A.t());
//        K.index_put_({Slice(0, m), m+n}, u);
//        K.index_put_({Slice(m, m+n), m+n+1}, v);

        Tensor zero_u = torch::zeros_like(u).unsqueeze(-1);
        Tensor zero_v = torch::zeros_like(v).unsqueeze(-1);
        Tensor eye_m = eye(M, OPTIONS);
        Tensor eye_n = eye(N, OPTIONS);

        Tensor K = cat(
            {
                cat({sigma * eye_m, -A,     u.unsqueeze(-1), zero_u}, 1),
                cat({-A.t(), sigma * eye_n, zero_v, v.unsqueeze(-1)}, 1)
            },
            0
        );
        Tensor c = torch::cat({phi, psi}, 0);

        // solve the underdetermined system
        Tensor x = std::get<0>(linalg_lstsq(K, c, nullopt, nullopt));

        // extract the solution, reverse pre-conditioning
        Tensor p = x.slice(0, 0, M) / SCALE;
        Tensor q = x.slice(0, M, M + N) / SCALE;
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
 * Tensor x = std::get<0>(linalg_lstsq(P, sigma*phi, nullopt, nullopt));
 * Tensor w = std::get<0>(linalg_lstsq(Q, A.t().mv(phi), nullopt, nullopt));
 * Tensor y = std::get<0>(linalg_lstsq(P, A.mv(psi), nullopt, nullopt));
 * Tensor z = std::get<0>(linalg_lstsq(Q, sigma*psi, nullopt, nullopt));
 * Tensor p = x + y;
 * Tensor q = w + z;

 * Tensor x, y, z, w;
 * if (phi_nonzero) {
 *     x = std::get<0>(linalg_lstsq(P, sigma*phi, nullopt, nullopt));
 *     w = std::get<0>(linalg_lstsq(Q, A.t().mv(phi), nullopt, nullopt));
 * } else {
 *     x = torch::zeros_like(phi);
 *     w = torch::zeros_like(psi);
 * }
 * if (psi_nonzero) {
 *     y = std::get<0>(linalg_lstsq(P, A.mv(psi), nullopt, nullopt));
 *     z = std::get<0>(linalg_lstsq(Q, sigma*psi, nullopt, nullopt));
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
    double atol = 1e-6,
    double rtol = 1e-6
) {
    /**
     * Wrap the struct into function.
     */
    auto output = SingularTriplet::apply(A, u0, v0, maxiter, atol, rtol);
    // assert(output.size() == 3);
    return std::make_tuple(output[0], output[1], output[2]);
}


TORCH_LIBRARY_FRAGMENT(liblinodenet, m) {
    m.def(
        "singular_triplet("
            "Tensor A,"
            "Tensor? u0=None,"
            "Tensor? v0=None,"
            "int? maxiter=None,"
            "float atol=1e-6,"
            "float rtol=1e-6"
        ") -> (Tensor, Tensor, Tensor)",
        singular_triplet
    );
}
