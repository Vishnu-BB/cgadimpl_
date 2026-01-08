#include "ad/ag_all.hpp"
#include <iostream>
#include <cmath>
#include <vector>

using namespace ag;
using namespace OwnTensor;

void check_gradients_valid(const std::string& name, const Value& v) {
    Tensor g = v.grad().to_cpu();
    const float* g_ptr = g.data<float>();
    for (size_t i = 0; i < g.numel(); ++i) {
        if (std::isnan(g_ptr[i]) || std::isinf(g_ptr[i])) {
            throw std::runtime_error("NaN or Inf detected in gradient of " + name);
        }
    }
}

int main() {
    std::cout << "=== Sparse Cross Entropy GPU Edge Cases Test ===\n";
    
    Device dev = Device::CUDA;

    // 1. Large Batch
    {
        std::cout << "Test 1: Large Batch (B=128, C=10)\n";
        const int B = 128;
        const int C = 10;
        Tensor Zt_cpu = OwnTensor::Tensor::randn(Shape{{B, C}}, TensorOptions().with_req_grad(true));
        Tensor Yt_cpu = OwnTensor::Tensor::zeros(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
        int64_t* y_ptr = Yt_cpu.data<int64_t>();
        for(int i=0; i<B; ++i) y_ptr[i] = i % C;
        
        Value Z = make_tensor(Zt_cpu.to_cuda(), "Z");
        Value Y = make_tensor(Yt_cpu.to_cuda(), "Y");
        Value loss = sparse_cross_entropy_with_logits(Z, Y);
        ag::backward(loss);
        check_gradients_valid("Z", Z);
        std::cout << "  Passed\n";
    }

    // 2. Single Batch
    {
        std::cout << "Test 2: Single Batch (B=1, C=5)\n";
        const int B = 1;
        const int C = 5;
        Tensor Zt_cpu = OwnTensor::Tensor::randn(Shape{{B, C}}, TensorOptions().with_req_grad(true));
        Tensor Yt_cpu = OwnTensor::Tensor::zeros(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
        Yt_cpu.data<int64_t>()[0] = 3;
        
        Value Z = make_tensor(Zt_cpu.to_cuda(), "Z");
        Value Y = make_tensor(Yt_cpu.to_cuda(), "Y");
        Value loss = sparse_cross_entropy_with_logits(Z, Y);
        ag::backward(loss);
        check_gradients_valid("Z", Z);
        std::cout << "  Passed\n";
    }

    // 3. High Confidence (Large Logits)
    {
        std::cout << "Test 3: High Confidence (Numerical Stability)\n";
        const int B = 2;
        const int C = 3;
        Tensor Zt_cpu = OwnTensor::Tensor::zeros(Shape{{B, C}}, TensorOptions().with_req_grad(true));
        float* z_ptr = Zt_cpu.data<float>();
        z_ptr[0] = 100.0f; z_ptr[1] = 0.0f; z_ptr[2] = 0.0f; // Target 0
        z_ptr[3] = 0.0f; z_ptr[4] = 100.0f; z_ptr[5] = 0.0f; // Target 1
        
        Tensor Yt_cpu = OwnTensor::Tensor::zeros(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
        Yt_cpu.data<int64_t>()[0] = 0;
        Yt_cpu.data<int64_t>()[1] = 1;
        
        Value Z = make_tensor(Zt_cpu.to_cuda(), "Z");
        Value Y = make_tensor(Yt_cpu.to_cuda(), "Y");
        Value loss = sparse_cross_entropy_with_logits(Z, Y);
        ag::backward(loss);
        check_gradients_valid("Z", Z);
        std::cout << "  Passed\n"; 
    }

    std::cout << "All GPU Edge Cases Passed!\n";
    return 0;
}
