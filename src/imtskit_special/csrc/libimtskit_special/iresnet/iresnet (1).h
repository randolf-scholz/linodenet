// iresnet.h
//
// Header for an invertible residual network block.  The block is
// implemented as a standard `torch::nn::Module` in C++ and exposes
// both a forward and an inverse method.  The forward method
// computes `y = x + f(x)` for a given residual transformation `f`,
// while the inverse solves the implicit fixed point equation
// `x = y - f(x)` using a simple fixed‑point iteration.  The
// gradient of the inverse is defined via a custom autograd
// Function which implements the inverse function theorem.  This
// header declares the module interface but does not depend on
// Python; the implementation is in `iresnet.cpp` and uses
// pybind11 to call back into Python for the user‑provided
// transformation.

#pragma once

#include <torch/torch.h>
#include <pybind11/pybind11.h>

// Forward declaration of the custom autograd function used in the
// inverse.  Defined in iresnet.cpp.
struct InverseFunction;

// An invertible residual block implemented as a Torch module.
//
// The template parameter allows the module to be wrapped by the
// `TORCH_MODULE` macro which creates a convenient alias type.  The
// constructor accepts a Python callable implementing the residual
// transformation `f: Tensor -> Tensor`, and optional parameters
// controlling the fixed‑point iteration used in the inverse.  Both
// forward and inverse operations operate batch‑wise on tensors of
// arbitrary dimensionality; all non‑batch dimensions are treated
// collectively as the feature dimension.
struct IResNetBlockImpl : public torch::nn::Module {
    using Tensor = torch::Tensor;

    // Construct an invertible residual block with the given
    // transformation.  `transformation` must be a Python callable
    // (for example, an `nn.Module` or any function) taking a
    // single tensor argument and returning a tensor of the same
    // shape.  `maxiter` and `tol` control the convergence criteria
    // for the fixed point solver used in the inverse.
    IResNetBlockImpl(pybind11::object transformation,
                     int64_t maxiter = 100,
                     double tol = 1e-3);

    // Perform the forward residual mapping y = x + f(x).  The
    // transformation is invoked under the Python GIL.
    Tensor forward(const Tensor& x);

    // Compute the inverse of the residual mapping using a custom
    // autograd function.  This solves x = y - f(x) and returns the
    // approximate solution.  The returned tensor participates in
    // autograd with a custom backward defined by the implicit
    // function theorem.
    Tensor inverse(const Tensor& y);

    // Python callable implementing the residual transformation.  We
    // store this as a pybind11 object so that the C++ code can
    // dispatch back into Python.  It is held here rather than in
    // static storage so that multiple blocks with different
    // transformations can coexist safely.
    pybind11::object transform_;

    // Maximum number of fixed‑point iterations used by the
    // inverse solver.
    int64_t maxiter_;

    // Convergence tolerance for the fixed‑point solver.
    double tol_;
};

// Define a convenient alias using the TORCH_MODULE macro.  This
// creates a type alias `IResNetBlock` corresponding to a
// `std::shared_ptr<IResNetBlockImpl>`, enabling idiomatic use in
// PyTorch C++ APIs.
TORCH_MODULE(IResNetBlock);
