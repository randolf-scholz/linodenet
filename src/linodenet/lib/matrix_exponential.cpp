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


struct MatrixExponential : public Function<MatrixExponential> {
    static std::vector<Tensor> forward(
        AutogradContext *ctx,
        const Tensor &A,
        const Tensor &dt,
        double atol = 1e-8,
        double rtol = 1e-5
    ) {
    /** @brief Compute the matrix exponential exp(A∆t) for multiple time steps.
     *
     * We make use of the Parallel Prefix Sum algorithm to compute in O(M³log N) instead of O(M³N).
     *
     * @see https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
     *
     * Matrix Exponential Implementation
     *
     * @see Computing the Matrix Exponential with an Optimized Taylor Polynomial Approximation.
     *      https://www.mdpi.com/2227-7390/7/12/1174
     *
     * @param ctx: context object
     * @param A: system matrix (shape: (M, M))
     * @param dt: time steps ∆t (shape: (..., N))
     * @param atol: absolute tolerance (default: 1e-8)
     * @param rtol: relative tolerance (default: 1e-5)
     * @returns exp(A∆t)  (shape: (..., N, M, M))
     *
     * @note We allow multiple time steps to be computed in parallel.
     * @note We allow batched inputs for dt.
     **/
        Tensor Adt = A * dt.unsqueeze(-2);
        Tensor expAdt = torch::linalg::matrix_exp(Adt);

        // After convergence, we have: Av = σu, Aᵀu = σv. Thus σ = uᵀAv.
        ctx->save_for_backward({A, dt});

        return {expAdt};
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
         */



        return { Tensor(), Tensor()};
    }
};




struct MatrixExponentialAction : public Function<MatrixExponential> {
    static std::vector<Tensor> forward(
        AutogradContext *ctx,
        const Tensor &A,
        const Tensor &dt,
        const Tensor &x,
        double atol = 1e-8,
        double rtol = 1e-5
    ) {
    /** @brief Compute action of the matrix exponential exp(A∆t).
     *
     * This function computes batched action of the matrix exponential
     *
     * (exp(A∆tₙ) xₙ)ₙ  (shape: (..., N, M)
     *
     * or:
     *
     * (exp(A∆tₙ) xₖ)ₙₖ  (shape: (..., N, K, M)
     *
     * @param ctx: context object
     * @param A: system matrix (shape: (M, M))
     * @param dt: time steps ∆t (shape: (..., N))
     * @param x: input vector (shape: (..., M))
     * @param atol: absolute tolerance (default: 1e-8)
     * @param rtol: relative tolerance (default: 1e-5)
     * @returns exp(A∆t)  (shape: (..., N, M))
     *
     * @note We allow multiple time steps to be computed in parallel.
     * @note We allow batched inputs for dt.
     **/

        // After convergence, we have: Av = σu, Aᵀu = σv. Thus σ = uᵀAv.
        ctx->save_for_backward({A, dt, x});

        return {A, dt, x};
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
         */
        static const torch::Tensor undef;
        return { Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
    }
};











// region export functions -------------------------------------------------------------
std::tuple<Tensor, Tensor, Tensor> matrix_exp(
        const Tensor &A,
        const Tensor &dt,
        double atol = 1e-8,
        double rtol = 1e-5
) {
    /**
     * Wrap the struct into function.
     */
    auto output = MatrixExponential::apply(A, dt, atol, rtol);
    // assert(output.size() == 3);
    return std::make_tuple(output[0], output[1], output[2]);
}

std::tuple<Tensor, Tensor, Tensor> matrix_exp_action(
        const Tensor &A,
        const Tensor &dt,
        double atol = 1e-8,
        double rtol = 1e-5
) {
    /**
     * Wrap the struct into function.
     */
    auto output = MatrixExponentialAction::apply(A, dt, atol, rtol);
    // assert(output.size() == 3);
    return std::make_tuple(output[0], output[1], output[2]);
}

TORCH_LIBRARY_FRAGMENT(liblinodenet, m) {
    m.def(
        "matrix_exp("
            "Tensor A,"
            "Tensor dt,"
            "float atol=1e-8,"
            "float rtol=1e-5"
        ") -> (Tensor, Tensor, Tensor)",
        matrix_exp
    );
    m.def(
        "matrix_exp("
            "Tensor A,"
            "Tensor dt,"
            "float atol=1e-8,"
            "float rtol=1e-5"
        ") -> (Tensor, Tensor, Tensor)",
        matrix_exp
    );
}
// endregion export functions ----------------------------------------------------------
