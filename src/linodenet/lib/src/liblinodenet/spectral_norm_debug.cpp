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
        double atol = 1e-6,
        double rtol = 1e-6
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

        // NOTE: during iteration u, u_old, v and v_old track the non-normalized vectors!!
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
        "spectral_norm_debug("
            "Tensor A,"
            "Tensor? u0=None,"
            "Tensor? v0=None,"
            "int? maxiter=None,"
            "float atol=1e-6,"
            "float rtol=1e-6"
        ") -> Tensor",
        spectral_norm_debug
    );
}
