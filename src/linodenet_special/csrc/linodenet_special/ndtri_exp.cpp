#include <torch/torch.h>
#include <array>
#include <limits>

using torch::Tensor;

namespace {
constexpr double UPPER_CUTOFF = -0.14541345786885906;  // log(1-e^-2)
constexpr double LOWER_CUTOFF = -2.0;
constexpr double SQRT_2 = 1.4142135623730951;

constexpr std::array<double, 9> P1 = {
    4.05544892305962419923,
    3.15251094599893866154e1,
    5.71628192246421288162e1,
    4.40805073893200834700e1,
    1.46849561928858024014e1,
    2.18663306850790267539,
    -1.40256079171354495875e-1,
    -3.50424626827848203418e-2,
    -8.57456785154685413611e-4,
};
constexpr std::array<double, 8> Q1 = {
    1.57799883256466749731e1,
    4.53907635128879210584e1,
    4.13172038254672030440e1,
    1.50425385692907503408e1,
    2.50464946208309415979,
    -1.42182922854787788574e-1,
    -3.80806407691578277194e-2,
    -9.33259480895457427372e-4,
};
constexpr std::array<double, 9> P2 = {
    3.23774891776946035970,
    6.91522889068984211695,
    3.93881025292474443415,
    1.33303460815807542389,
    2.01485389549179081538e-1,
    1.23716634817820021358e-2,
    3.01581553508235416007e-4,
    2.65806974686737550832e-6,
    6.23974539184983293730e-9,
};
constexpr std::array<double, 8> Q2 = {
    6.02427039364742014255,
    3.67983563856160859403,
    1.37702099489081330271,
    2.16236993594496635890e-1,
    1.34204006088543189037e-2,
    3.28014464682127739104e-4,
    2.89247864745380683936e-6,
    6.79019408009981274425e-9,
};

double finfo_min(at::ScalarType scalar_type) {
    switch (scalar_type) {
        case at::ScalarType::Half:
            return -65504.0;
        case at::ScalarType::BFloat16:
            return std::numeric_limits<float>::lowest();
        case at::ScalarType::Float:
            return std::numeric_limits<float>::lowest();
        case at::ScalarType::Double:
            return std::numeric_limits<double>::lowest();
        default:
            TORCH_CHECK(false, "ndtri_exp: unsupported dtype.");
    }
}

template <size_t N>
Tensor polevl(const Tensor &x, const std::array<double, N> &coeffs) {
    Tensor y = torch::zeros_like(x);
    for (double c : coeffs) {
        y = y * x + c;
    }
    return y;
}

template <size_t N>
Tensor p1evl(const Tensor &x, const std::array<double, N> &coeffs) {
    Tensor y = torch::ones_like(x);
    for (double c : coeffs) {
        y = y * x + c;
    }
    return y;
}

Tensor ndtri_exp_small(const Tensor &log_p) {
    const double finfo_min_value = finfo_min(log_p.scalar_type());
    const Tensor finfo_min_tensor = torch::full_like(log_p, finfo_min_value);

    const Tensor x = torch::where(
        log_p >= finfo_min_tensor * 0.5,
        torch::sqrt(-2 * log_p),
        SQRT_2 * torch::sqrt(-log_p)
    );
    const Tensor z = x.reciprocal();
    const Tensor x0 = x - z * x.log();
    const Tensor x1_small = z * polevl(z, P1) / p1evl(z, Q1);
    const Tensor x1_large = z * polevl(z, P2) / p1evl(z, Q2);
    const Tensor x1 = torch::where(x < 8.0, x1_small, x1_large);
    return x1 - x0;
}
} // namespace

static Tensor ndtri_exp_meta(const Tensor &log_p) {
    TORCH_CHECK(log_p.is_floating_point(), "ndtri_exp: log_p must be a floating point tensor.");
    return torch::empty_like(log_p);
}

static Tensor ndtri_exp(const Tensor &log_p) {
    TORCH_CHECK(log_p.is_floating_point(), "ndtri_exp: log_p must be a floating point tensor.");

    const double finfo_min_value = finfo_min(log_p.scalar_type());
    const Tensor finfo_min_tensor = torch::full_like(log_p, finfo_min_value);
    const Tensor neg_infinity = torch::full_like(log_p, -std::numeric_limits<double>::infinity());

    return torch::where(
        log_p < LOWER_CUTOFF,
        torch::where(
            log_p < finfo_min_tensor,
            neg_infinity,
            ndtri_exp_small(log_p)
        ),
        torch::where(
            log_p < UPPER_CUTOFF,
            torch::special::ndtri(log_p.exp()),
            -torch::special::ndtri(-log_p.expm1())
        )
    );
}

TORCH_LIBRARY_FRAGMENT(linodenet_special, m) {
    m.def("ndtri_exp(Tensor log_p) -> Tensor");
}

TORCH_LIBRARY_IMPL(linodenet_special, CompositeImplicitAutograd, m) {
    m.impl("ndtri_exp", &ndtri_exp);
}

TORCH_LIBRARY_IMPL(linodenet_special, Meta, m) {
    m.impl("ndtri_exp", &ndtri_exp_meta);
}
