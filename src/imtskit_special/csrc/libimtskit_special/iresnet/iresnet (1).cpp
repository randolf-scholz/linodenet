// iresnet.cpp
//
// Implementation of the invertible residual block defined in
// `iresnet.h`.  This file defines the forward, inverse and custom
// autograd behaviour for the block.  The inverse is computed via
// fixed‑point iteration and its gradient is implemented using a
// custom autograd::Function that applies the inverse function
// theorem.  Python callbacks are used to evaluate the user‑provided
// transformation; these are protected by the Python GIL.

#include "iresnet.h"

#include <torch/torch.h>
#include <torch/autograd.h>
#include <torch/linalg.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Forward declaration of the custom autograd function.  We define
// it below the block implementation to keep the public interface
// isolated in the header.
struct InverseFunction : public torch::autograd::Function<InverseFunction> {
    // Static storage for the Python transformation and its
    // parameters.  Because torch::autograd::Function enforces a
    // static interface, we cannot pass arbitrary state through the
    // function call.  Instead, we set these static members prior
    // to invoking `apply` from `IResNetBlockImpl::inverse`.  This
    // design is adequate for single‑threaded use; concurrent use
    // with different transformations would require a more
    // sophisticated context management.
    static py::object transform;
    static std::vector<torch::Tensor> parameters;
    static int64_t maxiter;
    static double tol;

    // Initialise the static fields.  This method must be called
    // before applying the function.  It acquires the Python GIL to
    // inspect the parameters of the Python transformation.  The
    // parameters are cached to allow gradient computation with
    // respect to the transformation's weights in the backward
    // method.
    static void setup(const py::object& t, int64_t maxiter_, double tol_) {
        transform = t;
        maxiter = maxiter_;
        tol = tol_;
        // Acquire GIL to iterate over Python parameters
        py::gil_scoped_acquire gil;
        parameters.clear();
        // The Python transformation is expected to have a
        // `parameters()` method returning an iterable of tensors.  We
        // copy these tensors into a vector so that we can pass
        // them to torch::autograd::grad in the backward method.
        py::object params_iter = transform.attr("parameters")();
        for (auto item : params_iter) {
            parameters.push_back(item.cast<torch::Tensor>());
        }
    }

    // Forward pass: solve x = y - f(x) by fixed‑point iteration.
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor y) {
        // Solve the fixed point using a simple iteration.  We run
        // this under a no‑grad guard to avoid constructing a
        // computation graph for the solver itself; gradients are
        // computed explicitly in the backward pass.  This reduces
        // memory usage compared to unrolling the iteration.
        torch::Tensor x;
        {
            torch::NoGradGuard no_grad;
            // Initialise x to zeros of the same shape as y.  One could
            // also initialise from y to accelerate convergence; we
            // choose zeros for simplicity.
            x = torch::zeros_like(y);
            torch::Tensor x_prev;
            for (int64_t i = 0; i < maxiter; ++i) {
                x_prev = x.clone();
                // Evaluate the transformation on x.  Acquire the GIL
                // before calling back into Python.
                {
                    py::gil_scoped_acquire gil;
                    torch::Tensor fx = transform(x).cast<torch::Tensor>();
                    x = y - fx;
                }
                // Check convergence by maximum absolute difference.
                double max_diff = (x - x_prev).abs().max().item<double>();
                if (max_diff < tol) {
                    break;
                }
            }
        }
        // Save x and y for backward.  We detach both to prevent
        // storing the solver graph.  They will be used only as
        // values in the backward pass.
        ctx->save_for_backward({x.detach(), y.detach()});
        return x;
    }

    // Backward pass: given ∂L/∂x (grad_output), compute ∂L/∂y and
    // accumulate gradients on the transformation parameters.  The
    // derivation follows the inverse function theorem for
    // deep equilibrium models.  See the accompanying Python
    // pseudocode for details.
    static torch::Tensor backward(torch::autograd::AutogradContext *ctx,
                                  torch::Tensor grad_output) {
        // Retrieve saved tensors.  x_saved is the fixed point and
        // y_saved is the original output of the residual block.
        auto saved = ctx->get_saved_variables();
        torch::Tensor x_saved = saved[0];
        torch::Tensor y_saved = saved[1];

        // Flatten grad_output so that we treat all non‑batch
        // dimensions as a single feature dimension.  This flattening
        // mirrors the flattening performed when constructing the
        // Jacobian and ensures consistent linear algebra.
        auto grad_view = grad_output.view({grad_output.size(0), -1});

        // Create a clone of x_saved that requires gradients.  We will
        // differentiate through f(x) with respect to both x and the
        // transformation parameters.  Note that we detach here to
        // avoid linking back into the solver graph; only the
        // function evaluation graph is needed.
        auto x = x_saved.detach().clone();
        x.set_requires_grad(true);

        // Compute f(x) and the residual g(x) = y - f(x).  Acquire
        // the GIL to call the Python transformation.  The result
        // `fx` participates in autograd and will be used to compute
        // gradients with respect to both x and the parameters.
        torch::Tensor fx;
        {
            py::gil_scoped_acquire gil;
            fx = transform(x).cast<torch::Tensor>();
        }
        torch::Tensor residual = y_saved - fx;

        // Flatten residual in the same way.  The residual and x have
        // the same shape, so this flattening is consistent.
        auto residual_flat = residual.view({residual.size(0), -1});

        const auto batch_size = residual_flat.size(0);
        const auto feature_dim = residual_flat.size(1);

        // Compute the Jacobian J = d(residual_flat)/d(x_flat).  We
        // allocate a (batch, feature_dim, feature_dim) tensor to
        // store the Jacobian.  Each slice J[b] is the Jacobian for
        // the b‑th batch element.
        auto jac = torch::zeros({batch_size, feature_dim, feature_dim}, x.options());

        // Compute each row of the Jacobian by differentiating the
        // residual with respect to x along a basis vector.  We use
        // torch::autograd::grad which mirrors Python's
        // torch.autograd.grad.  We set create_graph=false because we
        // do not need higher‑order derivatives inside the backward.
        for (int64_t i = 0; i < feature_dim; ++i) {
            // Build grad_output_i to pick out the i‑th output of
            // residual_flat.  It has ones in the i‑th column and
            // zeros elsewhere.
            auto grad_output_i = torch::zeros({batch_size, feature_dim}, x.options());
            grad_output_i.index_put_({torch::indexing::Slice(), i}, 1.0);
            // Compute the gradient of residual_flat with respect to x.
            auto grads = torch::autograd::grad({residual_flat}, {x}, {grad_output_i},
                                               /*retain_graph=*/true,
                                               /*create_graph=*/false);
            auto grad_r = grads[0].view({batch_size, -1});
            jac.index_put_({torch::indexing::Slice(), i, torch::indexing::Slice()}, grad_r);
        }

        // Form the linear system (I - Jᵀ) g = grad_view, where
        // J = d(residual)/d(x) = -d(f)/d(x).  This implies
        // I - Jᵀ = I + d(f)/d(x)ᵀ.  We solve for g which is
        // ∂L/∂y.  We perform batched solves by inverting each
        // matrix R[b] = I - J[b]ᵀ separately.  For moderate
        // dimensions this explicit inverse is acceptable.
        auto eye = torch::eye(feature_dim, x.options()).unsqueeze(0).expand({batch_size, feature_dim, feature_dim});
        auto R = eye - jac.transpose(1, 2);
        auto grad_rhs = grad_view.unsqueeze(-1);
        auto R_inv = torch::inverse(R);
        auto g = torch::matmul(R_inv, grad_rhs).squeeze(-1);

        // Reshape g back to the original gradient shape.  Compute the
        // number of trailing elements per feature for each batch.
        auto trailing = grad_output.numel() / grad_view.numel();
        torch::Tensor g_in_y;
        if (trailing > 1) {
            auto g_expanded = g.unsqueeze(-1).expand({batch_size, feature_dim, trailing});
            g_in_y = g_expanded.reshape_as(grad_output);
        } else {
            g_in_y = g.reshape_as(grad_output);
        }

        // Compute gradients with respect to the parameters of the
        // transformation.  The inverse function theorem yields
        // ∂L/∂p = -(gᵀ) * d(f(x))/d(p).  We achieve this by
        // differentiating f(x) with respect to each parameter using
        // autograd and grad_outputs = -g reshaped to the shape of
        // x.  This step ensures that the Python transformation's
        // parameters receive gradients during backpropagation through
        // the inverse.
        if (!parameters.empty()) {
            // Reshape g into the same shape as x so that it can
            // serve as grad_outputs for f(x).  Note that g was
            // computed in flattened form; reshape to match x.
            auto g_reshape = g.reshape_as(x);
            auto g_neg = -g_reshape;
            // Compute gradient of f(x) w.r.t each parameter.  We set
            // allow_unused=true because some parameters may not
            // influence f(x) for certain inputs.  retain_graph is
            // false since we no longer need the graph after this.
            auto grad_f_params = torch::autograd::grad({fx}, parameters, {g_neg},
                                                      /*retain_graph=*/false,
                                                      /*create_graph=*/false,
                                                      /*allow_unused=*/true);
            // Accumulate gradients into each parameter's .grad field.
            for (size_t i = 0; i < parameters.size(); ++i) {
                const torch::Tensor& grad_param = grad_f_params[i];
                if (!grad_param.defined()) {
                    continue;
                }
                auto& param = parameters[i];
                if (!param.grad().defined()) {
                    // If grad is uninitialised, clone the gradient.
                    param.mutable_grad() = grad_param.clone();
                } else {
                    param.grad().add_(grad_param);
                }
            }
        }

        // Return the gradient with respect to y.  autograd will
        // propagate this to upstream operators.  Only a single
        // tensor is returned because the only input to apply() is y.
        return g_in_y;
    }
};

// Define static member storage
py::object InverseFunction::transform;
std::vector<torch::Tensor> InverseFunction::parameters;
int64_t InverseFunction::maxiter = 100;
double InverseFunction::tol = 1e-3;

// IResNetBlockImpl constructor.  We store the Python callable and
// iteration parameters.  These are used in both forward and
// inverse methods.
IResNetBlockImpl::IResNetBlockImpl(py::object transformation,
                                   int64_t maxiter,
                                   double tol)
    : transform_(transformation), maxiter_(maxiter), tol_(tol) {
    // Nothing else to register: the transformation lives in Python
    // and exposes its parameters directly there.  In C++ we do not
    // register a submodule because pybind11 will manage parameter
    // traversal when computing gradients.
}

// Forward method: compute y = x + f(x).  Acquire the GIL while
// invoking the Python transformation.  We return the result of
// adding the input and transformation output.  This participates in
// autograd automatically.
torch::Tensor IResNetBlockImpl::forward(const torch::Tensor& x) {
    py::gil_scoped_acquire gil;
    auto fx = transform_(x).cast<torch::Tensor>();
    return x + fx;
}

// Inverse method: call the custom autograd function.  We first
// initialise the static context of the function with the current
// transformation and solver parameters, then apply the function to
// y.  The returned tensor has the same shape as y and will
// propagate gradients according to the implicit function theorem.
torch::Tensor IResNetBlockImpl::inverse(const torch::Tensor& y) {
    // Initialise static context.  A copy of the transformation's
    // parameters is captured at this time so that gradients can be
    // computed in the backward pass.  If the same module is used
    // concurrently from multiple threads with different
    // transformations, this design would need to be adapted to
    // thread‑local storage.
    InverseFunction::setup(transform_, maxiter_, tol_);
    // Apply the custom autograd function.  The only argument is y.
    auto result = InverseFunction::apply(y);
    return result;
}
