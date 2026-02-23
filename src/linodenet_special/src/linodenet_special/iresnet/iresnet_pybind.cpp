// iresnet_pybind.cpp
//
// Pybind11 binding that exposes an invertible residual block to Python.
// The wrapped class holds a Python callable representing the
// transformation f(x) and implements forward and inverse passes in
// C++, including a custom backward hook.  This allows a user to
// construct an invertible residual block from any Python function or
// nn.Module and use it as a drop‑in layer.
//
// Note: this binding relies on pybind11 and the PyTorch C++
// extension API.  To build the extension, see the accompanying
// setup.py.  Once compiled, you can import `iresnet_pybind` in
// Python and instantiate `IResNetBlockPy` with a callable.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <torch/indexing.h>
#include <iostream>

namespace py = pybind11;
using torch::indexing::Slice;

// A Python‑compatible invertible residual block.  Instead of storing
// a libtorch nn::Module, we store a Python callable.  This allows
// arbitrary Python functions or nn.Modules to be used as the
// transformation.  The implementation mirrors the logic of
// IResNetBlockImpl in iresnet.cpp but uses py::function for f.
class IResNetBlockPy {
public:
    // Construct with a Python callable and optional solver
    // parameters.  `tol` and `max_iters` are used for the fixed
    // point iteration in `inverse()`.
    IResNetBlockPy(py::function f, double tol = 1e-5, int max_iters = 50)
        : f_(std::move(f)), tol_(tol), max_iters_(max_iters) {}

    // Forward: y = x + f(x).  We acquire the GIL when calling
    // Python.  Autograd will handle gradients for the callable.
    torch::Tensor forward(const torch::Tensor& x) {
        py::gil_scoped_acquire gil;
        torch::Tensor fx = f_(x).cast<torch::Tensor>();
        return x + fx;
    }

    // Inverse: solve x = y - f(x) via fixed point iteration.  A
    // backward hook implements the gradient via the inverse
    // function theorem.  See iresnet.cpp for detailed comments.
    torch::Tensor inverse(const torch::Tensor& y,
                          double tol = -1.0,
                          int max_iters = -1) {
        if (tol < 0) tol = tol_;
        if (max_iters < 0) max_iters = max_iters_;
        // Fixed point solve in no‑grad scope
        torch::Tensor x_approx;
        {
            torch::NoGradGuard no_grad;
            x_approx = torch::zeros_like(y);
            torch::Tensor x_prev;
            for (int i = 0; i < max_iters; ++i) {
                x_prev = x_approx.clone();
                // Call f(x) under GIL
                torch::Tensor fx;
                {
                    py::gil_scoped_acquire gil;
                    fx = f_(x_approx).cast<torch::Tensor>();
                }
                x_approx = y - fx;
                double max_diff = (x_approx - x_prev).abs().max().item<double>();
                if (max_diff < tol) break;
            }
        }
        // Detach and require grad
        torch::Tensor x = x_approx.detach();
        x.set_requires_grad(true);
        // Clone y to avoid tracking history
        torch::Tensor y_clone = y.detach();
        // Register backward hook
        x.register_hook([this, y_clone, x](const torch::Tensor& grad) {
            // Flatten incoming gradient
            auto grad_view = grad.view({grad.size(0), -1});
            // Compute residual
            torch::Tensor fx;
            {
                py::gil_scoped_acquire gil;
                fx = f_(x).cast<torch::Tensor>();
            }
            auto residual = (y_clone - fx).view({y_clone.size(0), -1});
            auto batch_size = residual.size(0);
            auto feature_dim = residual.size(1);
            auto jac = torch::zeros({batch_size, feature_dim, feature_dim}, x.options());
            // Compute Jacobian row by row
            for (int64_t i = 0; i < feature_dim; ++i) {
                auto grad_output = torch::zeros({batch_size, feature_dim}, x.options());
                grad_output.index_put_({Slice(), i}, 1.0);
                // Compute gradient of residual w.r.t x
                auto grad_list = torch::autograd::grad({residual}, {x}, {grad_output},
                                                       /*retain_graph=*/true,
                                                       /*create_graph=*/true);
                auto grad_residual = grad_list[0].view({batch_size, -1});
                jac.index_put_({Slice(), i, Slice()}, grad_residual);
            }
            auto eye = torch::eye(feature_dim, x.options()).unsqueeze(0).expand({batch_size, feature_dim, feature_dim});
            auto R = eye - jac.transpose(1, 2);
            auto grad_rhs = grad_view.unsqueeze(-1);
            auto g = torch::matmul(torch::inverse(R), grad_rhs).squeeze(-1);
            // Reshape back to original gradient shape
            auto trailing = grad.numel() / grad_view.numel();
            if (trailing > 1) {
                auto g_expanded = g.unsqueeze(-1).expand({batch_size, feature_dim, trailing});
                return g_expanded.reshape_as(grad);
            }
            return g.reshape_as(grad);
        });
        return x;
    }

private:
    py::function f_;
    double tol_;
    int max_iters_;
};

PYBIND11_MODULE(iresnet_pybind, m) {
    py::class_<IResNetBlockPy>(m, "IResNetBlock")
        .def(py::init<py::function, double, int>(),
             py::arg("transformation"),
             py::arg("tol") = 1e-5,
             py::arg("max_iters") = 50)
        .def("forward", &IResNetBlockPy::forward)
        .def("inverse", &IResNetBlockPy::inverse,
             py::arg("y"),
             py::arg("tol") = -1.0,
             py::arg("max_iters") = -1);
}
