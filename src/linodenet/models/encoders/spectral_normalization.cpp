#include <c10/util/irange.h>
#include <torch/script.h>

#include <cstddef>
#include <string>

struct SpectralNormalization: public torch::autograd::Function<SpectralNormalization> {


    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor A,
        c10::optional<torch::Tensor> u0,
        c10::optional<torch::Tensor> v0,
        c10::optional<int64_t> maxiter,
        float32_t atol = 1e-8,
        float32_t rtol = 1e-5,
    ) {
        // initialize maxiter depending on the size of the matrix
        if (!maxiter.has_value()) {
            maxiter = std::max(A.size(0), A.size(1));
        }
        // Initialize u and v with random values if not given
        torch::Tensor u = u0.has_value() ? u0.value() : torch::randn({A.size(0)}, A.options());
        torch::Tensor v = v0.has_value() ? v0.value() : torch::randn({A.size(1)}, A.options());
        torch::Tensor u_old = u;
        torch::Tensor v_old = v;
        torch::Tensor sigma = torch.einsum("ij, i, j ->", A, u, v)


        // perform power-iteration for maxiter times or until convergence.
        for (const auto i : c10::irange(maxiter)) {
            u_old = u;
            v_old = v;

            u = torch::matmul(A, v_old);
            u /= torch::norm(u);

            v = torch::matmul(A.t(), u_old);
            v /= torch::norm(v);


            sigma = torch::dot(u, v)
// Formalizing as a optimization problem: min ‖A - σuvᵀ‖_F^2 s.t. ‖u‖₂ = ‖v‖₂ = 1
// The first order convergence criterion is ‖Av-σu‖₂ = 0 and ‖Aᵀu-σv‖₂ = 0
// Plugging in the iteration, we get ‖u' - σũ‖ = 0 and ‖v' - σṽ‖ = 0 (tilde indicates normalized vector)
// secondly we can estimate σ in each iteration via one of the 3 formulas
// (1) σ = uᵀAv  (2) σᵤ = ũᵀu'  (3) σᵥ = ṽᵀv'
// Plugging these into the equations we get
// ‖u' -  u'ᵀ ũᵀũ‖

            bool converged = (
                (torch::norm(torch::mv(A, v) - s * u) < atol * rtol * s
                && (torch::norm(torch::mv(A.t(), u) - s * v) < atol * rtol * s
            )

//            bool converged = (
//                (torch::norm(v - v_new) < atol + rtol * torch::norm(v))
//                && (torch::norm(u - u_new) < atol + rtol * torch::norm(u))
//            )
            if (converged) {break;}
        }
        // emit warning if no convergence within maxiter iterations
        if (i == maxiter - 1) {
            TORCH_WARN("Spectral norm estimation did not converge");
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
            torch::einsum("..., i, j -> ...ij", outer_grad, u, v);
        };
        return output;
    }
};

torch::Tensor spectral_normalization(
        torch::Tensor A,
        c10::optional<torch::Tensor> u0,
        c10::optional<torch::Tensor> v0,
        c10::optional<int64_t> maxiter,
        float32_t atol = 1e-8,
        float32_t rtol = 1e-5,
) {
    return SpectralNormalization::apply(A, u0, v0, maxiter, atol, rtol);
}

TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.def("spectral_normalization", spectral_normalization);
}
