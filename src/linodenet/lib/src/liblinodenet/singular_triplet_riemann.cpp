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
    /** @brief Compute the singular triplet of a matrix using Riemann Coordinate Descent. **/

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
        const int64_t MAXITER = maxiter ? maxiter.value() : 2*(M + N + 64);

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
        Tensor rho_u = torch::empty({}, OPTIONS);
        Tensor rho_v = torch::empty({}, OPTIONS);
        Tensor sigma_u = torch::empty({}, OPTIONS);
        Tensor sigma_v = torch::empty({}, OPTIONS);

        // Perform power-iteration for maxiter times or until convergence.
        // NOTE: Perform 2 iterations per loop to increase performance.
        //  Checking convergence is expensive, since `.item<bool>()` requires sync with CPU.
        //   The compiler cannot do this optimization on it's own because it would change behavior.
        for (auto i = 0; i<MAXITER; i++) {
            { // update u
                // compute gradient in ℝᵐ
                grad_u = A.mv(v);
                sigma_u = grad_u.dot(u);
                rho_u = grad_u.dot(grad_u) - sigma_u * sigma_u;
                // project gradient to tangent space 𝕋𝕊ᵐ
                grad_u -= sigma_u * u;
                // normalize tangent vector
                gamma_u = grad_u.norm();
                // update u = σᵤu + ρᵤg
                u = sigma_u * u + (rho_u/gamma_u) * grad_u;
                // u /= u.norm();  // not needed every iteration
            }
            { // update v
                // compute gradient in ℝⁿ
                grad_v = A_t.mv(u);
                sigma_v = grad_v.dot(v);
                rho_v = grad_v.dot(grad_v) - sigma_v * sigma_v;
                // project gradient to tangent space 𝕋𝕊ᵐ
                grad_v -= sigma_v * v;
                // normalize tangent vector
                gamma_v = grad_v.norm();
                // update v = σᵥv + ρᵥg
                v = sigma_v * v + (rho_v/gamma_v) * grad_v;
                // v /= v.norm();  // not needed every iteration
            }
            { // update u
                // compute gradient in ℝᵐ
                grad_u = A.mv(v);
                sigma_u = grad_u.dot(u);
                rho_u = grad_u.dot(grad_u) - sigma_u * sigma_u;
                // project gradient to tangent space 𝕋𝕊ᵐ
                grad_u -= sigma_u * u;
                // normalize tangent vector
                gamma_u = grad_u.norm();
                // update u = σᵤu + ρᵤg
                u = sigma_u * u + (rho_u/gamma_u) * grad_u;
                u /= u.norm();
            }
            { // update v
                // compute gradient in ℝⁿ
                grad_v = A_t.mv(u);
                sigma_v = grad_v.dot(v);
                rho_v = grad_v.dot(grad_v) - sigma_v * sigma_v;
                // project gradient to tangent space 𝕋𝕊ᵐ
                grad_v -= sigma_v * v;
                // normalize tangent vector
                gamma_v = grad_v.norm();
                // update v = σᵥv + ρᵥg
                v = sigma_v * v + (rho_v/gamma_v) * grad_v;
                v /= v.norm();
            }
            // check convergence
            // NOTE:(1/√2)(‖u‖+‖v‖) ≤ ‖(u,v)‖ ≤ ‖u‖+‖v‖ (via Jensen-Inequality)
            if ((converged = (  // compare against geometric mean
                (gamma_u + gamma_v) < ATOL + RTOL*torch::min(sigma_u, sigma_v)
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

        // solve the under-determined system
        Tensor x = std::get<0>(lstsq(K, c, nullopt, nullopt));

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


std::tuple<Tensor, Tensor, Tensor> singular_triplet_riemann(
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
        "singular_triplet_riemann("
            "Tensor A,"
            "Tensor? u0=None,"
            "Tensor? v0=None,"
            "int? maxiter=None,"
            "float atol=1e-6,"
            "float rtol=1e-6"
        ") -> (Tensor, Tensor, Tensor)",
        singular_triplet_riemann
    );
}
