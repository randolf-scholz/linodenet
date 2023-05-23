#include <torch/torch.h>

class MyModule : public torch::nn::Module {
public:
  MyModule(int64_t input_size, int64_t output_size)
      : linear(register_module("linear", torch::nn::Linear(input_size, output_size))) {}

  torch::Tensor forward(torch::Tensor x) {
    x = linear(x);
    return x;
  }

private:
  torch::nn::Linear linear{nullptr};
};
