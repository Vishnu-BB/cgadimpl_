#include "ad/ag_all.hpp"
#include <iostream>
#include <cmath>
#include <vector>

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
    std::cout << "=== Sparse Cross Entropy Hand-Checkable Test ===\n";
    
    // Batch=2, Classes=3
    const int B = 2;
    const int C = 3;

    // 1. Logits
    // [[1.0, 2.0, 3.0],
    //  [1.0, 0.0, -1.0]]
    Tensor Zt(Shape{{B, C}}, TensorOptions().with_req_grad(true));
    float* z_data = Zt.data<float>();
    z_data[0*C + 0] = 1.0f; z_data[0*C + 1] = 2.0f; z_data[0*C + 2] = 3.0f;
    z_data[1*C + 0] = 1.0f; z_data[1*C + 1] = 0.0f; z_data[1*C + 2] = -1.0f;
    Value Z = make_tensor(Zt, "Logits");
    print_val("Logits", Z);

    // 2. Targets (Indices)
    // [2, 0]
    // Note: Targets should be Int64 or Int32. Let's use Int64.
    Tensor Yt(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
    int64_t* y_data = Yt.data<int64_t>();
    y_data[0] = 2;
    y_data[1] = 0;
    Value Y = make_tensor(Yt, "Targets");
    
    // 3. Compute Loss
    Value loss = sparse_cross_entropy_with_logits(Z, Y);
    print_val("Loss", loss);

    // Expected Loss: 0.4075 (calculated previously)
    // Row 0: [1, 2, 3]. Max=3. Shift=[-2, -1, 0]. Exp=[0.1353, 0.3678, 1.0]. Sum=1.5031.
    // Softmax[2] = 1.0 / 1.5031 = 0.6653.
    // Loss = -log(0.6653) = 0.4075.
    
    // Row 1: [1, 0, -1]. Max=1. Shift=[0, -1, -2]. Exp=[1.0, 0.3678, 0.1353]. Sum=1.5031.
    // Softmax[0] = 1.0 / 1.5031 = 0.6653.
    // Loss = -log(0.6653) = 0.4075.
    
    // Mean Loss = 0.4075.

    // 4. Backward
    backward(loss);
    print_grad("Logits", Z);

    // Expected Gradients:
    // Scale = 1/2 = 0.5.
    // Row 0: Softmax=[0.0900, 0.2447, 0.6653]. Target=2 (OneHot=[0,0,1]).
    // Grad = (Softmax - OneHot) * 0.5
    //      = [0.0900, 0.2447, -0.3347] * 0.5
    //      = [0.0450, 0.1223, -0.1673]
    
    // Row 1: Softmax=[0.6653, 0.2447, 0.0900]. Target=0 (OneHot=[1,0,0]).
    // Grad = (Softmax - OneHot) * 0.5
    //      = [-0.3347, 0.2447, 0.0900] * 0.5
    //      = [-0.1673, 0.1223, 0.0450]

    // Verify values programmatically
    float expected_loss = 0.4075f;
    float actual_loss = loss.val().data<float>()[0];
    if (std::abs(actual_loss - expected_loss) > 1e-3) {
        std::cerr << "Loss mismatch! Expected " << expected_loss << ", got " << actual_loss << "\n";
        return 1;
    }

    float* g_ptr = Z.grad().data<float>();
    bool passed = true;
    
    // Row 0
    if (std::abs(g_ptr[0] - 0.0450f) > 1e-3) { std::cerr << "Grad[0,0] mismatch! Got " << g_ptr[0] << "\n"; passed = false; }
    if (std::abs(g_ptr[1] - 0.1223f) > 1e-3) { std::cerr << "Grad[0,1] mismatch! Got " << g_ptr[1] << "\n"; passed = false; }
    if (std::abs(g_ptr[2] - (-0.1673f)) > 1e-3) { std::cerr << "Grad[0,2] mismatch! Got " << g_ptr[2] << "\n"; passed = false; }
    
    // Row 1
    if (std::abs(g_ptr[3] - (-0.1673f)) > 1e-3) { std::cerr << "Grad[1,0] mismatch! Got " << g_ptr[3] << "\n"; passed = false; }
    if (std::abs(g_ptr[4] - 0.1223f) > 1e-3) { std::cerr << "Grad[1,1] mismatch! Got " << g_ptr[4] << "\n"; passed = false; }
    if (std::abs(g_ptr[5] - 0.0450f) > 1e-3) { std::cerr << "Grad[1,2] mismatch! Got " << g_ptr[5] << "\n"; passed = false; }

    if (passed) {
        std::cout << "Test Passed!\n";
        return 0;
    } else {
        std::cout << "Test Failed!\n";
        return 1;
    }
}
