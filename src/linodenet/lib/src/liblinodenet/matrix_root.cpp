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


// TensorFlow uses https://www.sciencedirect.com/science/article/pii/0024379587901182 (cf. https://www.tensorflow.org/api_docs/python/tf/linalg/sqrtm)
// MATLAB and SciPy use https://link.springer.com/chapter/10.1007/978-3-642-36803-5_12
// cf. https://www.mathworks.com/help/matlab/ref/sqrtm.html
// cf. https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html


struct MatrixRoot : public Function<MatrixRoot> {
    static std::vector<Tensor> forward(
        AutogradContext *ctx,
        const Tensor &A_in,
        const optional<Tensor> &u0,
        const optional<Tensor> &v0,
        optional<int64_t> maxiter,
        double atol = 1e-6,
        double rtol = 1e-6
    ) {
    }

    static variable_list backward(
        AutogradContext *ctx,
        const variable_list &grad_output
    ) {
        const auto saved = ctx->get_saved_variables();
    }
};


std::tuple<Tensor, Tensor, Tensor> matrix_root(
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
    auto output = MatrixRoot::apply(A, u0, v0, maxiter, atol, rtol);
    // assert(output.size() == 3);
    return std::make_tuple(output[0], output[1], output[2]);
}


TORCH_LIBRARY_FRAGMENT(liblinodenet, m) {
    m.def(
        "matrix_root("
            "Tensor A,"
            "Tensor? u0=None,"
            "Tensor? v0=None,"
            "int? maxiter=None,"
            "float atol=1e-6,"
            "float rtol=1e-6"
        ") -> (Tensor, Tensor, Tensor)",
        matrix_root
    );
}
