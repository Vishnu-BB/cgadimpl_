#include <iostream>
#include <vector>
#include <random>
#include "ad/ag_all.hpp"
#include "ad/utils/debug.hpp"

using namespace ag;

int main() {
    const int B = 8;
    const int In = 16;
    const int H1 = 64;
    const int H2 = 64;
    const int H3 = 32;
    const int H4 = 32;
    const int Out = 10;

    auto opts = TensorOptions();
    auto opts_param = TensorOptions().with_req_grad(true);

    Value X = make_tensor(Tensor::randn(Shape{{B, In}}, opts), "X");
    Value W1 = make_tensor(Tensor::randn(Shape{{In, H1}}, opts_param), "W1");
    Value b1 = make_tensor(Tensor::zeros(Shape{{1, H1}}, opts_param), "b1");
    Value W2 = make_tensor(Tensor::randn(Shape{{H1, H2}}, opts_param), "W2");
    Value b2 = make_tensor(Tensor::zeros(Shape{{1, H2}}, opts_param), "b2");
    Value W3 = make_tensor(Tensor::randn(Shape{{H2, H3}}, opts_param), "W3");
    Value b3 = make_tensor(Tensor::zeros(Shape{{1, H3}}, opts_param), "b3");
    Value W4 = make_tensor(Tensor::randn(Shape{{H3, H4}}, opts_param), "W4");
    Value b4 = make_tensor(Tensor::zeros(Shape{{1, H4}}, opts_param), "b4");
    Value W5 = make_tensor(Tensor::randn(Shape{{H4, Out}}, opts_param), "W5");
    Value b5 = make_tensor(Tensor::zeros(Shape{{1, Out}}, opts_param), "b5");

    std::cout << "Initial W1[0,0]: " << W1.val().to_cpu().data<float>()[0] << std::endl;

    Value L1 = gelu(matmul(X, W1) + b1);
    Value L2 = silu(matmul(L1, W2) + b2);
    Value L3 = leaky_relu(matmul(L2, W3) + b3, 0.1f);
    Value L4 = softplus(matmul(L3, W4) + b4);
    Value logits = matmul(L4, W5) + b5;

    Tensor Yt(Shape{{B, Out}}, opts);
    float* yt_data = Yt.data<float>();
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> pick(0, Out - 1);
    for (int i = 0; i < B; ++i) {
        int k = pick(gen);
        for (int j = 0; j < Out; ++j) {
            yt_data[i * Out + j] = (j == k) ? 1.0f : 0.0f;
        }
    }
    Value Y = make_tensor(Yt, "Y");

    Value loss = cross_entropy_with_logits(logits, Y);
    std::cout << "Loss: " << loss.val().to_cpu().data<float>()[0] << std::endl;

    zero_grad(loss);
    backward(loss);

    std::cout << "After backward W1[0,0]: " << W1.val().to_cpu().data<float>()[0] << std::endl;
    std::cout << "W1.grad[0,0]: " << W1.grad().to_cpu().data<float>()[0] << std::endl;

    return 0;
}
