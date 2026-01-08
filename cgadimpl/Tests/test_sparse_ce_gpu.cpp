#include "ad/ag_all.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <unordered_map>

using namespace ag;
using namespace OwnTensor;

// Helper to print a value
void print_val(const std::string& label, const Value& v) {
    std::cout << label << ":\n";
    v.val().display(std::cout, 4);
    std::cout << "\n";
}

void print_grad(const std::string& label, const Value& v) {
    std::cout << label << ".grad:\n";
    v.grad().display(std::cout, 4);
    std::cout << "\n";
}

int main() {
    std::cout << "=== Sparse Cross Entropy GPU Test ===\n";
    
    // Batch=2, Classes=3
    const int B = 2;
    const int C = 3;

    // 1. Logits
    // [[1.0, 2.0, 3.0],
    //  [1.0, 0.0, -1.0]]
    Tensor Zt_cpu = OwnTensor::Tensor::zeros(Shape{{B, C}}, TensorOptions().with_req_grad(true));
    std::vector<float> z_cpu = {1.0f, 2.0f, 3.0f, 1.0f, 0.0f, -1.0f};
    Zt_cpu.set_data(z_cpu);
    
    Tensor Zt = Zt_cpu.to_cuda();
    Value Z = make_tensor(Zt, "Logits");
    print_val("Logits", Z);

    // 2. Targets (Indices)
    // [2, 0]
    Tensor Yt_cpu = OwnTensor::Tensor::zeros(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
    std::vector<int64_t> y_cpu = {2, 0};
    Yt_cpu.set_data(y_cpu);
    
    Tensor Yt = Yt_cpu.to_cuda();
    Value Y = make_tensor(Yt, "Targets");
    
    // 3. Compute Loss
    Value loss = sparse_cross_entropy_with_logits(Z, Y);
    print_val("Loss", loss);

    // 4. Backward (VJP)
    ag::backward(loss);
    print_grad("Logits", Z);

    // 5. JVP Check
    // We'll use a tangent for Z: [[1, 0, 0], [0, 1, 0]]
    Tensor tZ_cpu = OwnTensor::Tensor::zeros(Shape{{B, C}}, TensorOptions());
    std::vector<float> tz_cpu_data = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    tZ_cpu.set_data(tz_cpu_data);
    Tensor tZ = tZ_cpu.to_cuda();
    
    // Compute JVP
    std::unordered_map<Node*, Tensor> seed;
    seed[Z.node.get()] = tZ;
    Tensor actual_jvp_t = ag::jvp(loss, seed);
    
    std::cout << "JVP:\n";
    actual_jvp_t.display(std::cout, 4);
    std::cout << "\n";

    // 6. Verification (Manual Calculation)
    float expected_loss = 0.4076f;
    float actual_loss = loss.val().to_cpu().data<float>()[0];
    
    std::cout << "Expected Loss: " << expected_loss << "\n";
    std::cout << "Actual Loss:   " << actual_loss << "\n";

    bool loss_ok = std::abs(actual_loss - expected_loss) < 1e-4;
    if (loss_ok) {
        std::cout << "Loss Check PASSED\n";
    } else {
        std::cout << "Loss Check FAILED\n";
    }

    std::vector<float> expected_grad = {
        0.0450f, 0.1224f, -0.1674f,
        -0.1674f, 0.1224f, 0.0450f
    };
    
    Tensor actual_grad_t = Z.grad().to_cpu();
    const float* actual_grad_ptr = actual_grad_t.data<float>();
    
    bool grad_ok = true;
    for (int i = 0; i < B * C; ++i) {
        if (std::abs(actual_grad_ptr[i] - expected_grad[i]) > 1e-4) {
            grad_ok = false;
            std::cout << "Grad mismatch at " << i << ": expected " << expected_grad[i] << ", got " << actual_grad_ptr[i] << "\n";
        }
    }

    if (grad_ok) {
        std::cout << "Gradient Check PASSED\n";
    } else {
        std::cout << "Gradient Check FAILED\n";
    }

    // JVP Verification
    // dL = (0.0450 * 1) + (0.1224 * 1) = 0.1674
    float expected_jvp = 0.1674f;
    float actual_jvp = actual_jvp_t.to_cpu().data<float>()[0];
    
    std::cout << "Expected JVP: " << expected_jvp << "\n";
    std::cout << "Actual JVP:   " << actual_jvp << "\n";

    bool jvp_ok = std::abs(actual_jvp - expected_jvp) < 1e-4;
    if (jvp_ok) {
        std::cout << "JVP Check PASSED\n";
    } else {
        std::cout << "JVP Check FAILED\n";
    }

    return (loss_ok && grad_ok && jvp_ok) ? 0 : 1;
}
