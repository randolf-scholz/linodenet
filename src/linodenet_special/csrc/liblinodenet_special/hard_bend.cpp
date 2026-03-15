#include "hard_bend.h"

namespace linodenet_special {
namespace {
void check_hard_bend_x(const Tensor &x) {
    TORCH_CHECK(x.is_floating_point(), "hard_bend: x must be a floating point tensor.");
}

Tensor hard_bend_impl(const Tensor &x, const Tensor &a, const Tensor &c, const Tensor &m) {
    const Tensor c_abs = c.abs();
    const Tensor m_signed = torch::copysign(m, a);
    const Tensor z = (a - m_signed) * x;
    return torch::where(z.abs() <= c_abs, a * x, m_signed * x + z.sign() * c_abs);
}
}  // namespace

Tensor hard_bend_meta(const Tensor &x, const Tensor &a, const Tensor &c, const Tensor &m) {
    check_hard_bend_x(x);
    TORCH_CHECK(a.is_floating_point(), "hard_bend: a must be a floating point tensor.");
    TORCH_CHECK(c.is_floating_point(), "hard_bend: c must be a floating point tensor.");
    TORCH_CHECK(m.is_floating_point(), "hard_bend: m must be a floating point tensor.");

    auto broadcasted = torch::broadcast_tensors({x, a, c, m});
    return torch::empty_like(broadcasted[0]);
}

/**
 * Piecewise linear function (3 regions), close the origin: a*x, outside: mx±c.
 *
 * @param x The input tensor to be activated.
 * @param a The slope of the middle region, defaults to ℯ².
 * @param c The offset of the parallel lines, defaults to 2.0.
 * @param m The slope of the outer regions, defaults to 1.0.
 * @return The transformed tensor after applying the hard bend activation.
 *
 * Note:
 *     An optimal transport from $N(0,1)$ to $½N(μ, σ²) + ½N(-μ, σ²)$ can be
 *     approximated with hard_bend(x, σℯ^{½μ²/σ²}, μ, σ).
 *     The defaults are chosen to approximate the optimal transport from
 *     $N(0,1)$ to $½N(2,1) + ½N(-2,1)$.
 *
 * Inversion formula: y = f(x, a, c, m) ⟺ x = f(y, 1/a, c, 1/m)
 */
Tensor hard_bend(const Tensor &x, const Tensor &a, const Tensor &c, const Tensor &m) {
    check_hard_bend_x(x);
    TORCH_CHECK(a.is_floating_point(), "hard_bend: a must be a floating point tensor.");
    TORCH_CHECK(c.is_floating_point(), "hard_bend: c must be a floating point tensor.");
    TORCH_CHECK(m.is_floating_point(), "hard_bend: m must be a floating point tensor.");

    auto broadcasted = torch::broadcast_tensors(
        {x, a.to(x.options()), c.to(x.options()), m.to(x.options())}
    );
    return hard_bend_impl(
        broadcasted[0],
        broadcasted[1],
        broadcasted[2],
        broadcasted[3]
    );
}
}  // namespace linodenet_special

TORCH_LIBRARY_FRAGMENT(linodenet_special, m) {
    m.def("hard_bend(Tensor x, Tensor a, Tensor c, Tensor m) -> Tensor");
}

TORCH_LIBRARY_IMPL(linodenet_special, CompositeImplicitAutograd, m) {
    m.impl("hard_bend", &linodenet_special::hard_bend);
}

TORCH_LIBRARY_IMPL(linodenet_special, Meta, m) {
    m.impl("hard_bend", &linodenet_special::hard_bend_meta);
}
