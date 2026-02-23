// iresnet_pybind.cpp
//
// Pybind11 bindings for the invertible residual block.  This module
// exposes a Python class `iResNetBlock` which wraps the C++
// implementation defined in iresnet.h and iresnet.cpp.  The
// resulting class behaves like a `torch.nn.Module` with a custom
// inverse method.  It accepts an arbitrary Python callable for
// the residual transformation and uses a fixed‑point solver and
// implicit differentiation implemented in C++.

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <torch/extension.h>

#include "iresnet.h"

namespace py = pybind11;

// Bindings entry point.  The module is named `iresnet_pybind`.
PYBIND11_MODULE(iresnet_pybind, m) {
    m.doc() = "Invertible residual network block implemented in C++";

    // Bind the C++ IResNetBlockImpl class as a Python class.  We
    // expose it under the name `iResNetBlock` to match the desired
    // Python API.  The class holds a Python callable for the
    // transformation and provides forward and inverse methods.
    py::class_<IResNetBlockImpl, std::shared_ptr<IResNetBlockImpl>>(m, "iResNetBlock")
        // Constructor: takes a Python callable, an integer maxiter
        // and a floating‑point tolerance.  Default values match the
        // user request (maxiter=100, tol=1e-3).
        .def(py::init<py::object, int64_t, double>(),
             py::arg("transformation"),
             py::arg("maxiter") = 100,
             py::arg("tol") = 1e-3,
             "Create an invertible residual block with the given transformation.")
        // Define __call__ to dispatch to forward, allowing the
        // instance to be called like a torch.nn.Module from Python.
        .def("__call__", &IResNetBlockImpl::forward,
             py::arg("x"),
             "Alias for forward(x)")
        // Forward pass: y = x + f(x).  This calls the stored
        // Python transformation under the hood.
        .def("forward", &IResNetBlockImpl::forward,
             py::arg("x"),
             "Compute the forward residual mapping y = x + f(x)")
        // Inverse pass: compute x from y by solving x = y - f(x)
        // using a fixed‑point iteration.  The returned tensor
        // carries gradients via a custom autograd Function.
        .def("inverse", &IResNetBlockImpl::inverse,
             py::arg("y"),
             "Compute the inverse of the residual mapping via fixed‑point iteration");
}
