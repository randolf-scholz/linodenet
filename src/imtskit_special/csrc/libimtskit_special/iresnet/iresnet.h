// iresnet.h
//
// This header defines a small invertible residual block for libtorch.
//
// The module exposed here implements a residual block of the form
// `y = x + f(x)` together with a custom inverse that solves the
// fixed‑point equation `x = y - f(x)` via simple fixed point iteration.
//
// A key feature of this implementation is that the inverse is
// differentiable: gradients flowing through the inverse are computed
// explicitly using the inverse function theorem.  Concretely, if
// `residual(x) = y - f(x)` denotes the implicit function used to
// recover `x`, then the Jacobian of `residual` with respect to `x`
// appears in the gradient formula.  During the backward pass we
// construct this Jacobian and solve a small linear system to obtain
// the gradient of the solution `x` with respect to the inputs `y`.
//
// For background on the necessary mathematics, the reader may
// consult the Deep Equilibrium Models tutorial, which shows how to
// differentiate through fixed points using the implicit function
// theorem.  The PyTorch C++ autograd API mirrors the Python API
// closely: functions such as `torch::autograd::grad`,
// `torch::Tensor::detach`, and `torch::Tensor::register_hook` are
// available to build custom gradients【739697034490799†L428-L437】.

#pragma once

#include <torch/torch.h>

// An invertible residual block.  Given a transformation module f, the
// forward pass computes y = x + f(x).  The inverse pass solves the
// implicit equation x = y - f(x) via fixed point iteration.  A custom
// backward hook is attached to the implicit solution so that gradients
// are computed by solving a linear system derived from the inverse
// function theorem.
//
// The transformation f can be any libtorch module with a `forward`
// method returning a tensor.  Because libtorch does not provide a
// common base class with a uniform `forward` signature, this class
// stores f using `torch::nn::AnyModule`, a type erased wrapper that
// allows polymorphic invocation of forward()【924145076684988†L121-L134】.

class IResNetBlockImpl : public torch::nn::Module {
public:
    // Construct an invertible residual block given a transformation
    // module.  The module is stored internally using type erasure and
    // registered as a submodule so that its parameters participate in
    // optimization.  A copy of the module is moved into the block.
    explicit IResNetBlockImpl(torch::nn::AnyModule transformation);

    // Forward computation.  Given an input tensor x, returns y = x + f(x).
    torch::Tensor forward(const torch::Tensor& x);

    // Inverse computation.  Given an output tensor y, solves for x such
    // that y = x + f(x) by fixed point iteration.  The optional
    // tolerance and maximum iteration parameters control the stopping
    // criterion.  The returned tensor requires gradients and is
    // equipped with a backward hook that implements the gradient
    // formula described above.
    torch::Tensor inverse(const torch::Tensor& y,
                          double tol = 1e-5,
                          int max_iters = 50);

private:
    // The wrapped transformation.  Stored via AnyModule so that
    // arbitrary modules may be passed at construction time.  When
    // calling forward(), the return type defaults to torch::Tensor.
    torch::nn::AnyModule transformation_;
};

// A convenient alias allowing users to write `IResNetBlock` instead of
// `IResNetBlockImpl` when declaring the module.
TORCH_MODULE(IResNetBlock);
