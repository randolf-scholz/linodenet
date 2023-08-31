#include <torch/script.h>

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


struct SpectralNorm: public Function<SpectralNorm> {
    static Tensor forward(
        AutogradContext *ctx,
        const Tensor &A,
        const optional<Tensor> &u0,
        const optional<Tensor> &v0,
        optional<int64_t> maxiter,
        double atol = 1e-8,
        double rtol = 1e-5
    ) {
        // Initialize maxiter depending on the size of the matrix.
        const auto A_t = A.t();
        const auto m = A.size(0);
        const auto n = A.size(1);
        const int64_t MAXITER = maxiter ? maxiter.value() : std::max<int64_t>(100, 2*(m + n));
        bool converged = false;

        // Initialize u and v with random values if not given
        Tensor u = u0 ? u0.value() : torch::randn({m}, A.options());
        Tensor v = v0 ? v0.value() : torch::randn({n}, A.options());

        // Initialize old values for convergence check
        // pre-allocate memory for residuals
        Tensor u_old = torch::empty_like(u);
        Tensor v_old = torch::empty_like(v);
        Tensor sigma_u = torch::empty({1}, u.options());
        Tensor sigma_v = torch::empty({1}, u.options());

        // NOTE: during iteration u, u_old, v and v_old track the unnormalized vectors!!
        // Perform power-iteration for maxiter times or until convergence.
        for (auto i = 0; i<MAXITER; i++) {
            // NOTE: We apply two iterations per loop. This is a case of duff's device.
            // This means that we effectively only check the stopping criterion every 2 iterations.
            // This improves performance on GPU since .item() requires a synchronization with CPU.
            // The compiler cannot do this optimization on it's own because it would change behavior.

            //; store previous values
            u_old = u;
            v_old = v;

            // update u
            sigma_v = v.norm();
            u = A.mv(v / sigma_v);

            // update v
            sigma_u = u.norm();
            v = A_t.mv(u / sigma_u);

            // check convergence
            if ((
                converged =
                (
                    ((u - u_old).norm() < atol + rtol*sigma_u)
                  & ((v - v_old).norm() < atol + rtol*sigma_v)
                ).item<bool>()
            )) {
                // Tensor sigma = A.mv(v).dot(u);
                // std::cout << at::str("Converged after ", i, " iterations. σ=", sigma) << std::endl;
                break;
            }
        }

        // Emit warning if no convergence within maxiter iterations.
        if (!converged) {
            TORCH_WARN("spectral_norm: no convergence in ", MAXITER, " iterations for input of shape ", A.sizes())
        }

        // store normalized vectors for backward pass
        sigma_u = u.norm();
        sigma_v = v.norm();
        u /= sigma_u;
        v /= sigma_v;
        ctx->save_for_backward({u, v});

        // compute and return sigma
        Tensor sigma = (sigma_u + sigma_v)/2;

        return sigma;
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


static inline Tensor spectral_norm_debug(
    const Tensor &A,
    const optional<Tensor> &u0,
    const optional<Tensor> &v0,
    optional<int64_t> maxiter,
    double atol = 1e-8,
    double rtol = 1e-5
) {
    /**
     * Wrap the struct into function.
     */
    return SpectralNorm::apply(A, u0, v0, maxiter, atol, rtol);
}


TORCH_LIBRARY_FRAGMENT(liblinodenet, m) {
    m.def(
        "spectral_norm_debug("
            "Tensor A,"
            "Tensor? u0=None,"
            "Tensor? v0=None,"
            "int? maxiter=None,"
            "float atol=1e-8,"
            "float rtol=1e-5"
        ") -> Tensor",
        spectral_norm_debug
    );
}




//using c10::optional;
//using torch::Tensor;
//
//static Tensor forward(
//    torch::autograd::AutogradContext *ctx,
//    const Tensor& A,
//    const optional<Tensor>& u0,
//    const optional<Tensor>& v0,
//    const optional<int64_t> maxiter,
//    const double atol = 1e-8,
//    const double rtol = 1e-5
//) {
//    // Initialize maxiter depending on the size of the matrix.
//    const auto m = A.size(0);
//    const auto n = A.size(1);
//    const int64_t MAXITER = maxiter.has_value() ? maxiter.value() : 4*(m + n);
//    bool converged = false;
//
//    // Initialize u and v with random values if not given
//    Tensor u = u0.has_value() ? u0.value() : torch::randn({m}, A.options());
//    Tensor v = v0.has_value() ? v0.value() : torch::randn({n}, A.options());
//    Tensor sigma = A.mv(v).dot(u);
//
//    // Perform power-iteration for maxiter times or until convergence.
//    // for (const auto i : c10::irange(MAXITER)) {
//    for (int64_t i = 0; i < MAXITER; ++i) {
//        Tensor u_old = u;
//        Tensor v_old = v;
//
//        u = A.mv(v);
//        sigma = dot(u, u_old);
//        Tensor left_residual = (u - sigma * u_old).norm();
//        u /= u.norm();
//        // assert(sigma.item().toDouble() > 0);  // TODO: is it clear this never happens?!
//
//        v = A.t().mv(u);
//        sigma = dot(v, v_old);
//        Tensor right_residual = (v - sigma * v_old).norm();
//        v /= v.norm();
//        // assert(sigma.item().toDouble() > 0);
//
//        Tensor tol = atol + rtol * sigma;
//        converged = (left_residual < tol).item<bool>() && (right_residual < tol).item<bool>();
//        if (converged) {
//            break;
//        }
//    }
//    // Emit warning if no convergence within maxiter iterations.
//    if (!converged) {
//        TORCH_WARN("Spectral norm estimation did not converge in ", MAXITER, " iterations.")
//    }
//    assert(sigma.item<double>() > 0);
//    // After convergence, we have: Av = σu, Aᵀu = σv. Thus σ = uᵀAv.
//    ctx->save_for_backward({u, v});
//    return sigma;
//}
