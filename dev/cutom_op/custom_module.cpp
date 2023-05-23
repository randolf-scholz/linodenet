#include <torch/torch.h>
// This header is all you need to do the C++ portions of this
// tutorial
#include <torch/script.h>
// This header is what defines the custom class registration
// behavior specifically. script.h already includes this, but
// we include it here so you know it exists in case you want
// to look at the API or implementation.
#include <torch/custom_class.h>

// https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html
#include <iostream>
#include <vector>

#include <torch/torch.h>
using namespace torch;
using namespace std;

struct AlexNetImpl : nn::Module
{

    AlexNetImpl(int64_t N)
        : conv1(register_module("conv1", nn::Conv2d(nn::Conv2dOptions(3, 64, 11).stride(4).padding(2)))),
          conv2(register_module("conv2", nn::Conv2d(nn::Conv2dOptions(64, 192, 5).padding(2)))),
          conv3(register_module("conv3", nn::Conv2d(nn::Conv2dOptions(192, 384, 3).padding(1)))),
          conv4(register_module("conv4", nn::Conv2d(nn::Conv2dOptions(384, 256, 3).padding(1)))),
          conv5(register_module("conv5", nn::Conv2d(nn::Conv2dOptions(256, 256, 3).padding(1)))),
          linear1(register_module("linear1", nn::Linear(9216, 4096))),
          linear2(register_module("linear2", nn::Linear(4096, 4096))),
          linear3(register_module("linear3", nn::Linear(4096, 1000))),
          dropout(register_module("dropout", nn::Dropout(nn::DropoutOptions(0.5)))) {}

    torch::Tensor forward(const torch::Tensor &input)
    {
        auto x = torch::relu(conv1(input));
        x = torch::max_pool2d(x, 3, 2);

        x = relu(conv2(x));
        x = max_pool2d(x, 3, 2);

        x = relu(conv3(x));
        x = relu(conv4(x));
        x = relu(conv5(x));
        x = max_pool2d(x, 3, 2);
        // Classifier, 256 * 6 * 6 = 9216
        x = x.view({x.size(0), 9216});
        x = dropout(x);
        x = relu(linear1(x));

        x = dropout(x);
        x = relu(linear2(x));

        x = linear3(x);
        return x;
    }
    torch::nn::Linear linear1, linear2, linear3;
    nn::Dropout dropout;
    nn::Conv2d conv1, conv2, conv3, conv4, conv5;
};

TORCH_MODULE_IMPL(AlexNet, AlexNetImpl);

// Define a custom module with forward and inverse methods.
// struct Net : torch::nn::Module
// {
//     Net() {}

//     torch::Tensor forward(torch::autograd::AutogradContext *ctx,
//                           torch::Tensor x)
//     {
//         // Implementation of the forward method.
//         ctx->save_for_backward({x});
//         return x * 2;
//     }

//     torch::Tensor inverse(torch::autograd::AutogradContext *ctx,
//                           torch::Tensor y)
//     {
//         // Implementation of the inverse method.
//         ctx->save_for_backward({y});
//         return y / 2;
//     }

//     // Override the backward function for the forward method.
//     torch::autograd::tensor_list forward_backward(
//         torch::autograd::AutogradContext *ctx,
//         torch::autograd::tensor_list grad_outputs)
//     {
//         auto saved = ctx->get_saved_variables();
//         auto x = saved[0];
//         auto grad_output = grad_outputs[0];
//         return {grad_output * 2};
//     }

//     // Override the backward function for the inverse method.
//     torch::autograd::tensor_list inverse_backward(
//         torch::autograd::AutogradContext *ctx,
//         torch::autograd::tensor_list grad_outputs)
//     {
//         auto saved = ctx->get_saved_variables();
//         auto y = saved[0];
//         auto grad_output = grad_outputs[0];
//         return {grad_output / 2};
//     }
// };

// int main()
// {
//     // Create an instance of the custom module.
//     CustomModule module;

//     // Create a tensor input to the forward method.
//     torch::Tensor x = torch::ones({2, 2}, torch::kFloat);

//     // Compute the forward method and backward pass.
//     auto y = module.forward(x);
//     auto grad_y = torch::ones_like(y);
//     y.backward(grad_y);

//     // Print the gradients of the forward method with respect to x.
//     std::cout << x.grad() << std::endl;

//     // Create a tensor input to the inverse method.
//     torch::Tensor z = torch::ones({2, 2}, torch::kFloat);

//     // Compute the inverse method and backward pass.
//     auto w = module.inverse(z);
//     auto grad_w = torch::ones_like(w);
//     w.backward(grad_w);

//     // Print the gradients of the inverse method with respect to z.
//     std::cout << z.grad() << std::endl;

//     return 0;
// }

TORCH_LIBRARY_FRAGMENT(custom_classes, m)
{
    m.class_<AlexNet>("AlexNet");
    // The following line registers the contructor of our MyStackClass
    // class that takes a single `std::vector<std::string>` argument,
    // i.e. it exposes the C++ method `MyStackClass(std::vector<T> init)`.
    // Currently, we do not support registering overloaded
    // constructors, so for now you can only `def()` one instance of
    // `torch::init`.
    // .def(torch::init())
    // The next line registers a stateless (i.e. no captures) C++ lambda
    // function as a method. Note that a lambda function must take a
    // `c10::intrusive_ptr<YourClass>` (or some const/ref version of that)
    // as the first argument. Other arguments can be whatever you want.
    // .def("forward", [](const c10::intrusive_ptr<MyStackClass<std::string>> &self)
    //      { return self->stack_.back(); })
    // The following four lines expose methods of the MyStackClass<std::string>
    // class as-is. `torch::class_` will automatically examine the
    // argument and return types of the passed-in method pointers and
    // expose these to Python and TorchScript accordingly. Finally, notice
    // that we must take the *address* of the fully-qualified method name,
    // i.e. use the unary `&` operator, due to C++ typing rules.
    // .def("forward", &Net::forward);
    // .def("pop", &MyStackClass<std::string>::pop)
    // .def("clone", &MyStackClass<std::string>::clone)
    // .def("merge", &MyStackClass<std::string>::merge);
}
