#include <torch/extension.h>
#include <vector>

//#include <ATen/ATen.h>
//#include <c10/util/irange.h>
//#include <torch/linalg.h>

//#include <cstddef>
//#include <string>


//import someLib as sl      ⟶  namespace sl = someLib;
//from someLib import func  ⟶  using someLib::func;
//from someLib import *     ⟶  using namespace someLib;

using c10::optional;
using torch::Tensor;
using torch::outer;
using torch::dot;
using torch::linalg::solve;

using c10::optional;
using torch::Tensor;



// s'(z) = (1 - s(z)) * s(z)
torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<torch::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);

  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
  auto gates = gate_weights.chunk(3, /*dim=*/1);

  auto input_gate = torch::sigmoid(gates[0]);
  auto output_gate = torch::sigmoid(gates[1]);
  auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = torch::tanh(new_cell) * output_gate;

  return {new_h,
          new_cell,
          input_gate,
          output_gate,
          candidate_cell,
          X,
          gate_weights};
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  auto d_output_gate = torch::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, /*dim=*/1);
  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
      torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights);
  const auto state_size = grad_h.size(1);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}







Tensor spectral_norm_forward(
    const Tensor& A,
    optional<Tensor> u0,
    optional<Tensor> v0,
    optional<int64_t> maxiter,
    const double atol = 1e-8,
    const double rtol = 1e-5
) {
    /** @brief Forward pass.
     *
     * @param ctx: context object
     * @param A: m x n matrix
     * @param u0: initial guess for left singular vector
     * @param v0: initial guess for right singular vector
     * @param maxiter: maximum number of iterations
     * @param atol: absolute tolerance
     * @param rtol: relative tolerance
     * @returns sigma: singular value
     */
    // Initialize maxiter depending on the size of the matrix.
    const auto m = A.size(0);
    const auto n = A.size(1);
    const int64_t MAXITER = maxiter.has_value() ? maxiter.value() : 4*(m + n);
    bool converged = false;

    // Initialize u and v with random values if not given
    Tensor u = u0.has_value() ? u0.value() : torch::randn({m}, A.options());
    Tensor v = v0.has_value() ? v0.value() : torch::randn({n}, A.options());
    Tensor sigma = A.mv(v).dot(u);

    // Perform power-iteration for maxiter times or until convergence.
    // for (const auto i : c10::irange(MAXITER)) {
    for (auto i=MAXITER; i--;) {
        Tensor u_old = u;
        Tensor v_old = v;

        u = A.mv(v);
        sigma = dot(u, u_old);
        Tensor left_residual = (u - sigma * u_old).norm();
        u /= u.norm();
        // assert(sigma.item().toDouble() > 0);  // TODO: is it clear this never happens?!

        v = A.t().mv(u);
        sigma = dot(v, v_old);
        Tensor right_residual = (v - sigma * v_old).norm();
        v /= v.norm();
        // assert(sigma.item().toDouble() > 0);

        Tensor tol = atol + rtol * sigma;
        converged = (left_residual < tol).item<bool>() && (right_residual < tol).item<bool>();
        if (converged) {
            break;
        }
    }
    // Emit warning if no convergence within maxiter iterations.
    if (!converged) {
        TORCH_WARN("Spectral norm estimation did not converge in ", MAXITER, " iterations.")
    }
    assert(sigma.item<double>() > 0);
    // After convergence, we have: Av = σu, Aᵀu = σv. Thus σ = uᵀAv.
    ctx->save_for_backward({u, v});
    return sigma;
}


static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output
) {
    /** @brief Backward Pass.
     *
     * Analytically, the VJP is ξ ↦ ξ⋅uvᵀ
     *
     * @param ctx: context object
     * @param grad_output: outer gradients
     * @returns g: gradient with respect to inputs
     */
    const auto saved = ctx->get_saved_variables();
    const auto u = saved[0];
    const auto v = saved[1];
    auto g_sigma = grad_output[0] * outer(u, v);
    return { g_sigma, Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
}







PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &spectral_norm_forward, "SN forward");
  m.def("backward", &spectral_norm_backward, "SN backward");
}
