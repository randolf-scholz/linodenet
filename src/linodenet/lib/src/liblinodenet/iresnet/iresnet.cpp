// iresnet.cpp
//
// Implementation of an invertible residual block in libtorch.  See
// iresnet.h for a description of the public API.  The key idea is
// that the inverse of a residual block can be obtained by solving a
// fixed point equation, and that the gradient of this inverse can be
// computed explicitly via the inverse function theorem.  This
// implementation follows the pseudocode shown in the accompanying
// Python example but uses only the C++ autograd API【739697034490799†L428-L437】.

#include "iresnet.h"

#include <torch/torch.h>
#include <torch/autograd.h>
#include <torch/nn/module.h>
#include <torch/linalg.h>
#include <torch/indexing.h>

#include <iostream>
#include <stdexcept>

using torch::indexing::Slice;

// Constructor: move the provided transformation into our AnyModule and
// register it so that its parameters are visible to optimizers.  The
// `AnyModule` wrapper allows us to accept arbitrary modules with a
// forward method and call them polymorphically【924145076684988†L121-L134】.
IResNetBlockImpl::IResNetBlockImpl(torch::nn::AnyModule transformation)
    : transformation_(std::move(transformation)) {
    // Register the underlying module with the base class so that
    // `parameters()` returns its parameters.  The `ptr()` method
    // provides a shared_ptr to the underlying module【924145076684988†L107-L110】.
    if (!transformation_.is_empty()) {
        // Register under name "f".  Note that we do not need to
        // register the AnyModule itself; instead we register the
        // contained module pointer so that parameter traversal works.
        this->register_module("f", transformation_.ptr());
    }
}

// Forward pass: compute y = x + f(x).  We rely on the fact that
// `AnyModule::forward` returns a torch::Tensor when no explicit
// template argument is provided【924145076684988†L121-L134】.
torch::Tensor IResNetBlockImpl::forward(const torch::Tensor& x) {
    // Evaluate the transformation on x.  Use .toTensor() to
    // convert AnyModule::Value to Tensor for older libtorch versions.
    auto fx_value = transformation_.forward(x);
    // Some versions of libtorch return AnyModule::Value which can be
    // implicitly converted to Tensor.  To be explicit, call
    // .toTensor() if available; otherwise rely on implicit
    // conversion.
    torch::Tensor fx;
    try {
        fx = fx_value.toTensor();
    } catch (const std::exception&) {
        // In case toTensor() is not available, attempt implicit
        // conversion via assignment.
        fx = fx_value;
    }
    return x + fx;
}

// Inverse pass: given y, solve the fixed point equation x = y - f(x).
// After computing an approximate solution, attach a backward hook
// implementing the gradient of the solution via the inverse
// function theorem.  The hook solves a linear system for each
// incoming gradient vector.  See the accompanying comments for
// details of each step.
torch::Tensor IResNetBlockImpl::inverse(const torch::Tensor& y,
                                         double tol,
                                         int max_iters) {
    // Verify input shape: the inverse operates on a 2‑D tensor of
    // shape (batch_size, feature_dim).  If additional dimensions are
    // present, they will be flattened implicitly by autograd::grad.
    if (y.dim() < 2) {
        throw std::invalid_argument("IResNetBlock::inverse expects a tensor"
                                    " of shape (batch, feature_dim) or higher");
    }

    // We will compute the fixed point using simple iteration inside
    // a no‑grad scope to avoid building a computation graph.  This
    // ensures that the approximate solution does not accumulate
    // unnecessary gradient history and that autograd sees x as a
    // leaf tensor when we later call `requires_grad_` on it【739697034490799†L340-L367】.
    torch::Tensor x_approx;
    {
        torch::NoGradGuard no_grad;
        // Initialize with zeros of the same shape as y.  One could
        // also start from y itself; both converge for contractive f.
        x_approx = torch::zeros_like(y);
        torch::Tensor x_prev;
        for (int i = 0; i < max_iters; ++i) {
            x_prev = x_approx.clone();
            // Compute f(x) using the stored module.  We call
            // `forward` through AnyModule and coerce to Tensor as
            // above.
            auto fx_value = transformation_.forward(x_approx);
            torch::Tensor fx;
            try {
                fx = fx_value.toTensor();
            } catch (const std::exception&) {
                fx = fx_value;
            }
            x_approx = y - fx;
            // Check for convergence by maximum absolute difference.
            double max_diff = (x_approx - x_prev).abs().max().item<double>();
            if (max_diff < tol) {
                break;
            }
        }
    }

    // Detach the approximate solution from the computation graph and
    // mark it as requiring gradients.  This will create a fresh leaf
    // tensor whose gradients we can customise via a hook【739697034490799†L340-L367】.
    torch::Tensor x = x_approx.detach();
    x.set_requires_grad(true);

    // Clone y so that the lambda captures a stable copy.  We call
    // detach() on y_clone to avoid accidentally retaining gradient
    // history through y.  y_clone has the same data as y but does
    // not require gradients.
    torch::Tensor y_clone = y.detach();

    // Register a backward hook on x.  When autograd computes the
    // gradient of some scalar loss L with respect to x (denoted
    // ∂L/∂x), this hook is invoked with `grad` = ∂L/∂x.  We return
    // the gradient with respect to y, ∂L/∂y, by solving the linear
    // system (I - Jᵀ) g = grad, where J is the Jacobian of
    // residual(x) = y - f(x) with respect to x.  See the deep
    // equilibrium tutorial for derivation.
    auto hook = [this, y_clone, x](const torch::Tensor& grad) {
        // grad has shape (batch_size, feature_dim, ...).  We flatten
        // trailing dimensions into the feature dimension so that the
        // Jacobian computation proceeds over two axes (batch and
        // feature).  For example, if x is (B, C, H, W) we reshape to
        // (B, N) with N = C * H * W.  This matches the assumption
        // that f maps from and to the same dimensionality.
        auto grad_view = grad.view({grad.size(0), -1});

        // Compute residual(x) = y - f(x) at the current fixed point.
        auto fx_value = transformation_.forward(x);
        torch::Tensor fx;
        try {
            fx = fx_value.toTensor();
        } catch (const std::exception&) {
            fx = fx_value;
        }
        auto residual = y_clone - fx;
        // Flatten residual in the same way.  We assume the first
        // dimension is the batch and the remaining dimensions are
        // features.
        residual = residual.view({residual.size(0), -1});

        // Determine shapes.
        const auto batch_size = residual.size(0);
        const auto feature_dim = residual.size(1);

        // Prepare storage for the Jacobian.  J[b, i, j] will hold
        // ∂residual[b, i] / ∂x[b, j].
        auto jac = torch::zeros({batch_size, feature_dim, feature_dim}, x.options());

        // We need to compute the Jacobian by repeated calls to
        // torch::autograd::grad.  The C++ API does not expose
        // torch.autograd.functional.jacobian, so we explicitly loop
        // over the feature dimension.  The approach mirrors the
        // Python code but uses the C++ `grad` function【739697034490799†L428-L437】.
        for (int64_t i = 0; i < feature_dim; ++i) {
            // Create grad_output with ones in the i‑th column and zeros
            // elsewhere.  This selects the i‑th component of the
            // residual for differentiation.  Because residual has
            // shape (batch, feature_dim), grad_output has the same
            // shape.
            auto grad_output = torch::zeros({batch_size, feature_dim}, x.options());
            // Assign 1 to the i‑th column for all batch entries.
            grad_output.index_put_({Slice(), i}, 1.0);

            // Compute gradient of residual with respect to x.  We set
            // retain_graph = true so that subsequent calls do not
            // delete the graph, and create_graph = true so that the
            // resulting gradient itself records operations.  The
            // latter is important because the gradient depends on
            // parameters of f, and thus further differentiation
            // through this hook should be possible.
            auto grad_list = torch::autograd::grad({residual}, {x}, {grad_output},
                                                   /*retain_graph=*/true,
                                                   /*create_graph=*/true);
            auto grad_residual = grad_list[0];
            // Flatten the gradient to match the shape (batch, feature_dim).
            grad_residual = grad_residual.view({batch_size, -1});
            // Write into the Jacobian.  The slice along dimension 1
            // corresponds to the i‑th row of the Jacobian.
            jac.index_put_({Slice(), i, Slice()}, grad_residual);
        }

        // Form the linear system (I - Jᵀ) g = grad_view.  Note that
        // J has shape (B, feat, feat).  We transpose the last two
        // dimensions to obtain Jᵀ (for each batch), subtract from
        // the identity, and then invert.  Instead of using
        // torch::linalg::solve, which may not be present in all
        // versions, we call torch::inverse followed by matmul.  For
        // small feature dimensions this is acceptable.
        auto eye = torch::eye(feature_dim, x.options()).unsqueeze(0).expand({batch_size, feature_dim, feature_dim});
        auto R = eye - jac.transpose(1, 2);
        // Solve the system.  Add an extra dimension to grad_view to
        // perform batched matrix multiplication.  The result g has
        // shape (batch, feature_dim).
        auto grad_rhs = grad_view.unsqueeze(-1);
        auto R_inv = torch::inverse(R);
        auto g = torch::matmul(R_inv, grad_rhs).squeeze(-1);

        // Reshape g back to the shape of grad.  Because we flattened
        // spatial dimensions into feature_dim, we need to view g into
        // the same shape as the incoming gradient.  grad_view is
        // (batch, feature_dim); the original grad may have additional
        // dimensions after the first two.  Compute the number of
        // trailing elements per batch.
        auto trailing = grad.numel() / grad_view.numel();
        if (trailing > 1) {
            // Expand g to (batch, feature_dim, trailing) and then
            // reshape to the original grad shape.
            auto g_expanded = g.unsqueeze(-1).expand({batch_size, feature_dim, trailing});
            return g_expanded.reshape_as(grad);
        }
        return g.reshape_as(grad);
    };

    // Attach the hook.  register_hook returns a handle which would
    // allow removal of the hook if desired.  Storing the handle
    // prevents premature destruction of the hook, but since x is
    // returned and lives until the backward pass finishes, it is
    // sufficient to ignore the return value here【739697034490799†L588-L598】.
    x.register_hook(hook);

    return x;
}
