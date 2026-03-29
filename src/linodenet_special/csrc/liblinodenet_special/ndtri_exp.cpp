#include "ndtri_exp.h"

#include <array>
#include <limits>
#include <mutex>
#include <vector>

namespace linodenet_special {
namespace {
constexpr double UPPER_CUTOFF = -0.14541345786885906;  // log(1-e^-2)
constexpr double LOWER_CUTOFF = -2.0;
constexpr double SQRT_2 = 1.4142135623730951;
constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();

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

struct CoeffCacheKey {
    c10::DeviceType device_type;
    c10::DeviceIndex device_index;
    at::ScalarType scalar_type;

    friend bool operator==(const CoeffCacheKey &lhs, const CoeffCacheKey &rhs) {
        return (
            lhs.device_type == rhs.device_type &&
            lhs.device_index == rhs.device_index &&
            lhs.scalar_type == rhs.scalar_type
        );
    }
};

struct CoeffTensors {
    Tensor p1;
    Tensor q1;
    Tensor p2;
    Tensor q2;
};

double finfo_min(const at::ScalarType &scalar_type) {
    return AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, scalar_type, "finfo_min",
        [&] {
            return static_cast<double>(std::numeric_limits<scalar_t>::lowest());
        }
    );
}

CoeffTensors get_coeffs(const Tensor &x) {
    static std::mutex cache_mutex;
    static std::vector<std::pair<CoeffCacheKey, CoeffTensors>> cache;

    const auto device = x.device();
    const CoeffCacheKey key{device.type(), device.index(), x.scalar_type()};

    {
        // Fast path: return immediately when this device/dtype combination was
        // already materialized by an earlier call.
        const std::lock_guard lock(cache_mutex);
        for (const auto &[cached_key, coeffs] : cache) {
            if (cached_key == key) {
                return coeffs;
            }
        }
    }

    // Build the coefficient tensors outside the mutex. Tensor construction can
    // be relatively expensive, so we do not want unrelated cache lookups to
    // block on this work.
    const auto options = x.options();
    CoeffTensors coeffs{
        torch::tensor(std::vector(P1.begin(), P1.end()), options),
        torch::tensor(std::vector(Q1.begin(), Q1.end()), options),
        torch::tensor(std::vector(P2.begin(), P2.end()), options),
        torch::tensor(std::vector(Q2.begin(), Q2.end()), options),
    };

    const std::lock_guard lock(cache_mutex);
    // Another thread may have inserted the same key while we were constructing
    // `coeffs`, so re-check before appending a new cache entry.
    for (const auto &[cached_key, cached_coeffs] : cache) {
        if (cached_key == key) {
            return cached_coeffs;
        }
    }
    cache.emplace_back(key, coeffs);
    return coeffs;
}

Tensor polyeval8(const Tensor &x, const Tensor &coeffs) {
    Tensor y = torch::zeros_like(x);
    y = at::addcmul(coeffs[0], x, y);
    y = at::addcmul(coeffs[1], x, y);
    y = at::addcmul(coeffs[2], x, y);
    y = at::addcmul(coeffs[3], x, y);
    y = at::addcmul(coeffs[4], x, y);
    y = at::addcmul(coeffs[5], x, y);
    y = at::addcmul(coeffs[6], x, y);
    y = at::addcmul(coeffs[7], x, y);
    y = at::addcmul(coeffs[8], x, y);
    return y;
}

Tensor poly1eval8(const Tensor &x, const Tensor &coeffs) {
    Tensor y = torch::ones_like(x);
    y = at::addcmul(coeffs[0], x, y);
    y = at::addcmul(coeffs[1], x, y);
    y = at::addcmul(coeffs[2], x, y);
    y = at::addcmul(coeffs[3], x, y);
    y = at::addcmul(coeffs[4], x, y);
    y = at::addcmul(coeffs[5], x, y);
    y = at::addcmul(coeffs[6], x, y);
    y = at::addcmul(coeffs[7], x, y);
    return y;
}

Tensor ndtri_exp_small(const Tensor &log_p) {
    const auto [p1, q1, p2, q2] = get_coeffs(log_p);

    const Tensor x = torch::sqrt(-2.0 * log_p);
    const Tensor z = x.reciprocal();
    const Tensor x0 = at::addcmul(x, z, x.log(), -1.0);
    const Tensor x1_small = z * polyeval8(z, p1) / poly1eval8(z, q1);
    const Tensor x1_large = z * polyeval8(z, p2) / poly1eval8(z, q2);
    const Tensor x1 = torch::where(x < 8.0, x1_small, x1_large);
    return x1 - x0;
}
}  // namespace

Tensor ndtri_exp_meta(const Tensor &log_p) {
    TORCH_CHECK(log_p.is_floating_point(), "ndtri_exp: log_p must be a floating point tensor.");
    return torch::empty_like(log_p);
}

Tensor ndtri_exp(const Tensor &log_p) {
    const Tensor invalid_mask = log_p.isnan() | (log_p > 0.0);
    const Tensor neginf_mask = log_p.isneginf();
    const Tensor small_mask = (log_p < LOWER_CUTOFF) & ~(invalid_mask | neginf_mask);
    const Tensor medium_mask = (log_p >= LOWER_CUTOFF) & (log_p <= UPPER_CUTOFF);
    const Tensor large_mask = (log_p > UPPER_CUTOFF) & ~invalid_mask;

    // Mask the unused part of each branch with a safe dummy value.
    // This prevents propagation of spurious NANs through inactive branches.
    const Tensor dummy = torch::scalar_tensor(-1.0, log_p.options());
    const Tensor small_input = torch::where(small_mask, log_p, dummy);
    const Tensor medium_input = torch::where(medium_mask, log_p, dummy);
    const Tensor large_input = torch::where(large_mask, log_p, dummy);

    const Tensor neg_infinity = torch::scalar_tensor(NEG_INFINITY, log_p.options());
    const Tensor invalid = torch::full({}, std::numeric_limits<double>::quiet_NaN(), log_p.options());
    const Tensor small = ndtri_exp_small(small_input);
    const Tensor medium = torch::special::ndtri(medium_input.exp());
    const Tensor large = -torch::special::ndtri(-large_input.expm1());

    return torch::where(
        invalid_mask,
        invalid,
        torch::where(
            neginf_mask,
            neg_infinity,
            torch::where(
                small_mask,
                small,
                torch::where(medium_mask, medium, large)
            )
        )
    );
}
}  // namespace linodenet_special

TORCH_LIBRARY_FRAGMENT(linodenet_special, m) {
    m.def("ndtri_exp(Tensor log_p) -> Tensor");
}

TORCH_LIBRARY_IMPL(linodenet_special, CompositeImplicitAutograd, m) {
    m.impl("ndtri_exp", &linodenet_special::ndtri_exp);
}

TORCH_LIBRARY_IMPL(linodenet_special, Meta, m) {
    m.impl("ndtri_exp", &linodenet_special::ndtri_exp_meta);
}
