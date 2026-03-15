#include <torch/torch.h>








TORCH_LIBRARY_FRAGMENT(linodenet_special, m) {
    m.def("bimodal_to_gaussian(Tensor _, Tensor mu, Tensor sigma) -> Tensor");
    m.def("gaussian_to_bimodal(Tensor _, Tensor mu, Tensor sigma) -> Tensor");
    m.def("mixture_to_gaussian(Tensor _, Tensor weights, Tensor mus, Tensor sigmas) -> Tensor");
    m.def("gaussian_to_mixture(Tensor _, Tensor weights, Tensor mus, Tensor sigmas) -> Tensor");
}

TORCH_LIBRARY_IMPL(linodenet_special, Autograd, m) {
    m.impl("bimodal_to_gaussian", &bimodal_to_gaussian_impl);
    m.impl("gaussian_to_bimodal", &gaussian_to_bimodal_impl);
    m.impl("mixture_to_gaussian", &mixture_to_gaussian_impl);
    m.impl("gaussian_to_mixture", &gaussian_to_mixture_impl);
}

TORCH_LIBRARY_IMPL(linodenet_special, Meta, m) {
    m.impl("bimodal_to_gaussian", &bimodal_to_gaussian_meta);
    m.impl("gaussian_to_bimodal", &gaussian_to_bimodal_meta);
    m.impl("mixture_to_gaussian", &mixture_to_gaussian_meta);
    m.impl("gaussian_to_mixture", &gaussian_to_mixture_meta);
}
