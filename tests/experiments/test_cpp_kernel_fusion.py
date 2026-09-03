import torch
from torch.utils.benchmark import Timer
from torch.utils.cpp_extension import load_inline

NUM_THREADS = 1
torch.set_num_threads(NUM_THREADS)

# language=C++
CPP_SOURCE = r"""
#include <ATen/ATen.h>
#include <torch/library.h>

at::Tensor abxy(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& x,
    const at::Tensor& y
) {
    // Deliberately naive:
    //
    //   1. aten::mul(a, b)
    //   2. aten::mul(x, y)
    //   3. aten::add(...)
    //
    // On CUDA eager execution this means three CUDA kernels.
    const at::Tensor ab = at::mul(a, b);
    const at::Tensor xy = at::mul(x, y);
    return at::add(ab, xy);
}

TORCH_LIBRARY(abxy_ext, m) {
    m.def("abxy(Tensor a, Tensor b, Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(abxy_ext, CompositeImplicitAutograd, m) {
    m.impl("abxy", TORCH_FN(abxy));
}
"""

# language=C++
CUDA_CPP_SOURCE = r"""
#include <ATen/ATen.h>
#include <torch/library.h>


at::Tensor abxy_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& x,
    const at::Tensor& y
);


TORCH_LIBRARY(abxy_cuda_ext, m) {
    m.def("abxy(Tensor a, Tensor b, Tensor x, Tensor y) -> Tensor");
}


TORCH_LIBRARY_IMPL(abxy_cuda_ext, CUDA, m) {
    m.impl("abxy", TORCH_FN(abxy_cuda));
}
"""

# language=CU
CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>


__global__ void abxy_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    const int64_t size
) {
    const auto index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        out[index] = fmaf(a[index], b[index], x[index] * y[index]);
    }
}


at::Tensor abxy_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& x,
    const at::Tensor& y
) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == at::kFloat, "a must have dtype float32");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_cuda() && b.device() == a.device(), "b must match a's device");
    TORCH_CHECK(x.is_cuda() && x.device() == a.device(), "x must match a's device");
    TORCH_CHECK(y.is_cuda() && y.device() == a.device(), "y must match a's device");
    TORCH_CHECK(
        b.scalar_type() == at::kFloat
            && x.scalar_type() == at::kFloat
            && y.scalar_type() == at::kFloat,
        "all tensors must have dtype float32"
    );
    TORCH_CHECK(
        b.sizes() == a.sizes()
            && x.sizes() == a.sizes()
            && y.sizes() == a.sizes(),
        "all tensors must have the same shape"
    );
    TORCH_CHECK(
        b.is_contiguous() && x.is_contiguous() && y.is_contiguous(),
        "all tensors must be contiguous"
    );

    const c10::cuda::CUDAGuard device_guard(a.device());
    const auto out = at::empty_like(a);
    const auto size = a.numel();
    if (size == 0) {
        return out;
    }

    constexpr int threads = 256;
    const auto blocks = static_cast<int>((size + threads - 1) / threads);
    abxy_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""


# Compile and load the C++ code.
#
# is_python_module=False means there is no pybind module. Loading the shared
# library merely executes TORCH_LIBRARY and registers abxy_ext::abxy with
# PyTorch's dispatcher.
load_inline(
    name="abxy_ext",
    cpp_sources=CPP_SOURCE,
    extra_cflags=["-O3"],
    is_python_module=False,
    verbose=False,
)

load_inline(
    name="abxy_cuda_ext",
    cpp_sources=CUDA_CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    is_python_module=False,
    verbose=False,
)


@torch.library.register_fake("abxy_cuda_ext::abxy")  # pyright: ignore[reportUntypedFunctionDecorator]
def _(
    a: torch.Tensor,
    _b: torch.Tensor,
    _x: torch.Tensor,
    _y: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(a)


# --------------------------------------------------------------------------------------
# Implementations
# --------------------------------------------------------------------------------------


def python_fn(
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return a * b + x * y


def cpp_fn(
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.abxy_ext.abxy(a, b, x, y)


def cuda_fn(
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.abxy_cuda_ext.abxy(a, b, x, y)


def optimized_fn(
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return torch.addcmul(a * b, x, y)


python_compiled = torch.compile(
    python_fn,
    fullgraph=True,
    backend="inductor",
)

cpp_compiled = torch.compile(
    cpp_fn,
    fullgraph=True,
    backend="inductor",
)

optimized_compiled = torch.compile(
    optimized_fn,
    fullgraph=True,
    backend="inductor",
)

cuda_compiled = torch.compile(
    cuda_fn,
    fullgraph=True,
    backend="inductor",
)

# Uncomment this BEFORE the first compiled calls if you want to see the
# generated Inductor kernels:
#
# torch._logging.set_logs(kernel_code=True)


# --------------------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------------------


def benchmark(size: int, repeats: int, batch_size: int) -> None:
    args = tuple(
        torch.randn(batch_size, size, device="cuda", dtype=torch.float32)
        for _ in range(4)
    )
    a, b, x, y = args

    functions = {
        "Python": (python_fn, python_compiled),
        "C++": (cpp_fn, cpp_compiled),
        "CUDA": (cuda_fn, cuda_compiled),
        "Optimized": (optimized_fn, optimized_compiled),
    }

    for uncompiled_fn, compiled_fn in functions.values():
        uncompiled_fn(*args)
        compiled_fn(*args)
    torch.cuda.synchronize()

    # Correctness.
    expected = python_fn(*args)
    for uncompiled_fn, compiled_fn in functions.values():
        torch.testing.assert_close(uncompiled_fn(*args), expected)
        torch.testing.assert_close(compiled_fn(*args), expected)

    print(
        f"\nBatch size = {batch_size:,}; tensor size = {size:,}; repeats = {repeats:,}"
    )
    print(f"{'':20s} {'Uncompiled':>12s} {'Compiled':>12s}")

    for name, (uncompiled_fn, compiled_fn) in functions.items():
        timings = []
        for fn in (uncompiled_fn, compiled_fn):
            fn(*args)
            torch.cuda.synchronize()
            bench = Timer(
                stmt="fn(a, b, x, y)",
                globals={
                    "fn": fn,
                    "a": a,
                    "b": b,
                    "x": x,
                    "y": y,
                },
                num_threads=NUM_THREADS,
            )

            with torch.compiler.set_stance("fail_on_recompile"):
                result = bench.timeit(repeats)

            timings.append(result.median * 1e6)

        print(f"{name:20s} {timings[0]:9.2f} us {timings[1]:9.2f} us")


if __name__ == "__main__":
    print("PyTorch:", torch.__version__)
    print("CUDA:   ", torch.version.cuda)
    print("GPU:    ", torch.cuda.get_device_name())

    for size, repeats, batch_size in [
        (4, 100_000, 128),
        (64, 100_000, 128),
        (1_024, 10_000, 128),
        (8_192, 10_000, 128),
    ]:
        benchmark(size, repeats, batch_size)
