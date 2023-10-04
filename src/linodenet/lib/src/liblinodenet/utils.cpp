#include <torch/script.h>

namespace linodenet {
namespace utils {
// lookup table for the tolerance for different floating point types
constexpr double toleranceLookup(at::ScalarType scalarType) {
    switch (scalarType) {
        // regular floats
        case at::ScalarType::Half: return 1e-3;
        case at::ScalarType::Float: return 1e-6;
        case at::ScalarType::Double: return 1e-15;
        // complex floats
        case at::ScalarType::ComplexHalf: return 1e-3;
        case at::ScalarType::ComplexFloat: return 1e-6;
        case at::ScalarType::ComplexDouble: return 1e-15;
        // other floats
        case at::ScalarType::BFloat16: return 1e-2;
        default: throw std::invalid_argument("Unsupported ScalarType");
    }
}

inline bool allclose(
        at::Tensor & input,
        at::Tensor & other,
        c10::optional<double> rtol,
        c10::optional<double> atol,
        bool equal_nan
) {
    double tol = toleranceLookup(input.scalar_type());
    double a = atol ? atol.value() : tol;
    double r = rtol ? rtol.value() : tol;
    return torch::allclose(input, other, r, a, equal_nan);
}

inline at::Tensor isclose(
        at::Tensor & input,
        at::Tensor & other,
        c10::optional<double> rtol,
        c10::optional<double> atol,
        bool equal_nan
) {
    double tol = toleranceLookup(input.scalar_type());
    double a = atol ? atol.value() : tol;
    double r = rtol ? rtol.value() : tol;
    return torch::isclose(input, other, r, a, equal_nan);
}

inline bool close(
        at::Tensor & input,
        at::Tensor & other,
        c10::optional<double> rtol,
        c10::optional<double> atol
) {
    double tol = toleranceLookup(input.scalar_type());
    double a = atol ? atol.value() : tol;
    double r = rtol ? rtol.value() : tol;
    return ((input - other).norm() <= a + r * other.norm()).item<bool>();
}
} // namespace utils
} // namespace liblinodenet
