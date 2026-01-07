#include "ad/ag_all.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <limits>

using namespace ag;
using namespace OwnTensor;

// Helper to print test section
void print_section(const std::string& name) {
    std::cout << "\n========================================\n";
    std::cout << name << "\n";
    std::cout << "========================================\n";
}

// Helper to check if gradients are valid (no NaN, no Inf)
bool check_gradients_valid(const Value& v) {
    const float* g_ptr = v.grad().data<float>();
    int64_t numel = v.grad().numel();
    for (int64_t i = 0; i < numel; ++i) {
        if (std::isnan(g_ptr[i]) || std::isinf(g_ptr[i])) {
            std::cerr << "Invalid gradient at index " << i << ": " << g_ptr[i] << "\n";
            return false;
        }
    }
    return true;
}

// Test 1: Single class (edge case - no real choice)
bool test_single_class() {
    print_section("Test 1: Single Class");
    
    const int B = 3;
    const int C = 1;
    
    Tensor Zt(Shape{{B, C}}, TensorOptions().with_req_grad(true));
    float* z_data = Zt.data<float>();
    z_data[0] = 5.0f;
    z_data[1] = -2.0f;
    z_data[2] = 0.0f;
    Value Z = make_tensor(Zt, "Logits");
    
    Tensor Yt(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
    int64_t* y_data = Yt.data<int64_t>();
    y_data[0] = 0;
    y_data[1] = 0;
    y_data[2] = 0;
    Value Y = make_tensor(Yt, "Targets");
    
    Value loss = sparse_cross_entropy_with_logits(Z, Y);
    backward(loss);
    
    // With single class, softmax is always 1.0, so loss should be 0
    float actual_loss = loss.val().data<float>()[0];
    std::cout << "Loss: " << actual_loss << " (expected ~0.0)\n";
    
    if (!check_gradients_valid(Z)) return false;
    
    // Gradient should be 0 (softmax - one_hot = 1 - 1 = 0)
    float* g_ptr = Z.grad().data<float>();
    for (int i = 0; i < B; ++i) {
        if (std::abs(g_ptr[i]) > 1e-5) {
            std::cerr << "Gradient should be ~0, got " << g_ptr[i] << "\n";
            return false;
        }
    }
    
    std::cout << "✓ Single class test passed\n";
    return true;
}

// Test 2: Large number of classes
bool test_many_classes() {
    print_section("Test 2: Many Classes (1000)");
    
    const int B = 2;
    const int C = 1000;
    
    Tensor Zt(Shape{{B, C}}, TensorOptions().with_req_grad(true));
    float* z_data = Zt.data<float>();
    for (int i = 0; i < B * C; ++i) {
        z_data[i] = (i % 10) * 0.1f; // Small varied values
    }
    Value Z = make_tensor(Zt, "Logits");
    
    Tensor Yt(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
    int64_t* y_data = Yt.data<int64_t>();
    y_data[0] = 0;
    y_data[1] = 999;
    Value Y = make_tensor(Yt, "Targets");
    
    Value loss = sparse_cross_entropy_with_logits(Z, Y);
    backward(loss);
    
    if (!check_gradients_valid(Z)) return false;
    
    // Check gradient sum per row should be ~0 (softmax sums to 1, one_hot sums to 1)
    float* g_ptr = Z.grad().data<float>();
    for (int b = 0; b < B; ++b) {
        float row_sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            row_sum += g_ptr[b * C + c];
        }
        if (std::abs(row_sum) > 1e-3) {
            std::cerr << "Row " << b << " gradient sum should be ~0, got " << row_sum << "\n";
            return false;
        }
    }
    
    std::cout << "✓ Many classes test passed\n";
    return true;
}

// Test 3: Extreme logit values (numerical stability)
bool test_extreme_logits() {
    print_section("Test 3: Extreme Logit Values");
    
    const int B = 4;
    const int C = 3;
    
    Tensor Zt(Shape{{B, C}}, TensorOptions().with_req_grad(true));
    float* z_data = Zt.data<float>();
    
    // Row 0: Very large positive values
    z_data[0*C + 0] = 100.0f; z_data[0*C + 1] = 50.0f; z_data[0*C + 2] = 80.0f;
    
    // Row 1: Very large negative values
    z_data[1*C + 0] = -100.0f; z_data[1*C + 1] = -50.0f; z_data[1*C + 2] = -80.0f;
    
    // Row 2: Mixed extreme values
    z_data[2*C + 0] = 100.0f; z_data[2*C + 1] = -100.0f; z_data[2*C + 2] = 0.0f;
    
    // Row 3: All same (uniform distribution)
    z_data[3*C + 0] = 5.0f; z_data[3*C + 1] = 5.0f; z_data[3*C + 2] = 5.0f;
    
    Value Z = make_tensor(Zt, "Logits");
    
    Tensor Yt(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
    int64_t* y_data = Yt.data<int64_t>();
    y_data[0] = 0;
    y_data[1] = 1;
    y_data[2] = 0;
    y_data[3] = 1;
    Value Y = make_tensor(Yt, "Targets");
    
    Value loss = sparse_cross_entropy_with_logits(Z, Y);
    backward(loss);
    
    float actual_loss = loss.val().data<float>()[0];
    std::cout << "Loss: " << actual_loss << "\n";
    
    if (!check_gradients_valid(Z)) return false;
    
    // Row 0: Target is max, should have very small loss
    // Row 3: Uniform distribution, loss should be -log(1/3) ≈ 1.0986
    
    std::cout << "✓ Extreme logits test passed\n";
    return true;
}

// Test 4: Large batch size
bool test_large_batch() {
    print_section("Test 4: Large Batch Size");
    
    const int B = 1000;
    const int C = 10;
    
    Tensor Zt(Shape{{B, C}}, TensorOptions().with_req_grad(true));
    float* z_data = Zt.data<float>();
    for (int i = 0; i < B * C; ++i) {
        z_data[i] = ((i * 7) % 13) * 0.5f; // Pseudo-random values
    }
    Value Z = make_tensor(Zt, "Logits");
    
    Tensor Yt(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
    int64_t* y_data = Yt.data<int64_t>();
    for (int i = 0; i < B; ++i) {
        y_data[i] = i % C; // Cycle through classes
    }
    Value Y = make_tensor(Yt, "Targets");
    
    Value loss = sparse_cross_entropy_with_logits(Z, Y);
    backward(loss);
    
    if (!check_gradients_valid(Z)) return false;
    
    std::cout << "Loss: " << loss.val().data<float>()[0] << "\n";
    std::cout << "✓ Large batch test passed\n";
    return true;
}

// Test 5: All targets point to same class
bool test_same_target_class() {
    print_section("Test 5: All Targets Same Class");
    
    const int B = 5;
    const int C = 4;
    
    Tensor Zt(Shape{{B, C}}, TensorOptions().with_req_grad(true));
    float* z_data = Zt.data<float>();
    for (int i = 0; i < B * C; ++i) {
        z_data[i] = (i % 7) * 0.3f;
    }
    Value Z = make_tensor(Zt, "Logits");
    
    Tensor Yt(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
    int64_t* y_data = Yt.data<int64_t>();
    for (int i = 0; i < B; ++i) {
        y_data[i] = 2; // All point to class 2
    }
    Value Y = make_tensor(Yt, "Targets");
    
    Value loss = sparse_cross_entropy_with_logits(Z, Y);
    backward(loss);
    
    if (!check_gradients_valid(Z)) return false;
    
    std::cout << "Loss: " << loss.val().data<float>()[0] << "\n";
    std::cout << "✓ Same target class test passed\n";
    return true;
}

// Test 6: Gradient accumulation (multiple backward passes)
bool test_gradient_accumulation() {
    print_section("Test 6: Gradient Accumulation");
    
    const int B = 2;
    const int C = 3;
    
    Tensor Zt(Shape{{B, C}}, TensorOptions().with_req_grad(true));
    float* z_data = Zt.data<float>();
    z_data[0*C + 0] = 1.0f; z_data[0*C + 1] = 2.0f; z_data[0*C + 2] = 3.0f;
    z_data[1*C + 0] = 1.0f; z_data[1*C + 1] = 0.0f; z_data[1*C + 2] = -1.0f;
    Value Z = make_tensor(Zt, "Logits");
    
    Tensor Yt(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
    int64_t* y_data = Yt.data<int64_t>();
    y_data[0] = 2;
    y_data[1] = 0;
    Value Y = make_tensor(Yt, "Targets");
    
    // First backward
    Value loss1 = sparse_cross_entropy_with_logits(Z, Y);
    backward(loss1);
    
    float* g_ptr = Z.grad().data<float>();
    float first_grad = g_ptr[0];
    
    // Second backward (should accumulate)
    Value loss2 = sparse_cross_entropy_with_logits(Z, Y);
    backward(loss2);
    
    float second_grad = g_ptr[0];
    
    // Second gradient should be approximately 2x first gradient
    if (std::abs(second_grad - 2.0f * first_grad) > 1e-3) {
        std::cerr << "Gradient accumulation failed. First: " << first_grad 
                  << ", Second: " << second_grad << " (expected ~" << 2.0f * first_grad << ")\n";
        return false;
    }
    
    std::cout << "✓ Gradient accumulation test passed\n";
    return true;
}

// Test 7: Zero logits
bool test_zero_logits() {
    print_section("Test 7: All Zero Logits");
    
    const int B = 3;
    const int C = 4;
    
    Tensor Zt(Shape{{B, C}}, TensorOptions().with_req_grad(true));
    float* z_data = Zt.data<float>();
    for (int i = 0; i < B * C; ++i) {
        z_data[i] = 0.0f; // All zeros
    }
    Value Z = make_tensor(Zt, "Logits");
    
    Tensor Yt(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
    int64_t* y_data = Yt.data<int64_t>();
    y_data[0] = 0;
    y_data[1] = 1;
    y_data[2] = 3;
    Value Y = make_tensor(Yt, "Targets");
    
    Value loss = sparse_cross_entropy_with_logits(Z, Y);
    backward(loss);
    
    // With all zeros, softmax is uniform: 1/C for each class
    // Loss should be -log(1/C) = log(C)
    float expected_loss = std::log(static_cast<float>(C));
    float actual_loss = loss.val().data<float>()[0];
    
    std::cout << "Loss: " << actual_loss << " (expected ~" << expected_loss << ")\n";
    
    if (std::abs(actual_loss - expected_loss) > 1e-3) {
        std::cerr << "Loss mismatch!\n";
        return false;
    }
    
    if (!check_gradients_valid(Z)) return false;
    
    std::cout << "✓ Zero logits test passed\n";
    return true;
}

// Test 8: Int32 targets (alternative dtype)
bool test_int32_targets() {
    print_section("Test 8: Int32 Targets");
    
    const int B = 2;
    const int C = 3;
    
    Tensor Zt(Shape{{B, C}}, TensorOptions().with_req_grad(true));
    float* z_data = Zt.data<float>();
    z_data[0*C + 0] = 1.0f; z_data[0*C + 1] = 2.0f; z_data[0*C + 2] = 3.0f;
    z_data[1*C + 0] = 1.0f; z_data[1*C + 1] = 0.0f; z_data[1*C + 2] = -1.0f;
    Value Z = make_tensor(Zt, "Logits");
    
    // Use Int32 instead of Int64
    Tensor Yt(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int32));
    int32_t* y_data = Yt.data<int32_t>();
    y_data[0] = 2;
    y_data[1] = 0;
    Value Y = make_tensor(Yt, "Targets");
    
    Value loss = sparse_cross_entropy_with_logits(Z, Y);
    backward(loss);
    
    float expected_loss = 0.4075f;
    float actual_loss = loss.val().data<float>()[0];
    
    if (std::abs(actual_loss - expected_loss) > 1e-3) {
        std::cerr << "Loss mismatch! Expected " << expected_loss << ", got " << actual_loss << "\n";
        return false;
    }
    
    if (!check_gradients_valid(Z)) return false;
    
    std::cout << "✓ Int32 targets test passed\n";
    return true;
}

// Test 9: Gradient magnitude check
bool test_gradient_magnitude() {
    print_section("Test 9: Gradient Magnitude Bounds");
    
    const int B = 10;
    const int C = 5;
    
    Tensor Zt(Shape{{B, C}}, TensorOptions().with_req_grad(true));
    float* z_data = Zt.data<float>();
    for (int i = 0; i < B * C; ++i) {
        z_data[i] = ((i * 13) % 17) * 0.2f - 1.5f; // Range roughly [-1.5, 1.9]
    }
    Value Z = make_tensor(Zt, "Logits");
    
    Tensor Yt(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
    int64_t* y_data = Yt.data<int64_t>();
    for (int i = 0; i < B; ++i) {
        y_data[i] = i % C;
    }
    Value Y = make_tensor(Yt, "Targets");
    
    Value loss = sparse_cross_entropy_with_logits(Z, Y);
    backward(loss);
    
    // Gradients should be bounded: (softmax - one_hot) / batch_size
    // softmax is in [0, 1], one_hot is in {0, 1}
    // So gradient should be in [-1/B, 1/B]
    float max_expected_grad = 1.0f / B + 1e-3; // Small epsilon for numerical errors
    
    float* g_ptr = Z.grad().data<float>();
    for (int i = 0; i < B * C; ++i) {
        if (std::abs(g_ptr[i]) > max_expected_grad) {
            std::cerr << "Gradient magnitude too large at index " << i 
                      << ": " << g_ptr[i] << " (max expected: " << max_expected_grad << ")\n";
            return false;
        }
    }
    
    std::cout << "✓ Gradient magnitude test passed\n";
    return true;
}

// Test 10: Consistency with one-hot version
bool test_consistency_with_onehot() {
    print_section("Test 10: Consistency with One-Hot CE");
    
    const int B = 3;
    const int C = 4;
    
    // Create logits
    Tensor Zt(Shape{{B, C}}, TensorOptions().with_req_grad(true));
    float* z_data = Zt.data<float>();
    z_data[0*C + 0] = 1.5f; z_data[0*C + 1] = 2.3f; z_data[0*C + 2] = 0.8f; z_data[0*C + 3] = 1.1f;
    z_data[1*C + 0] = 0.5f; z_data[1*C + 1] = 1.2f; z_data[1*C + 2] = 2.1f; z_data[1*C + 3] = 0.3f;
    z_data[2*C + 0] = 1.8f; z_data[2*C + 1] = 0.9f; z_data[2*C + 2] = 1.4f; z_data[2*C + 3] = 2.0f;
    Value Z_sparse = make_tensor(Zt, "Logits_Sparse");
    
    // Create sparse targets
    Tensor Yt_sparse(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
    int64_t* y_sparse_data = Yt_sparse.data<int64_t>();
    y_sparse_data[0] = 1;
    y_sparse_data[1] = 2;
    y_sparse_data[2] = 3;
    Value Y_sparse = make_tensor(Yt_sparse, "Targets_Sparse");
    
    // Compute sparse CE
    Value loss_sparse = sparse_cross_entropy_with_logits(Z_sparse, Y_sparse);
    backward(loss_sparse);
    
    // Create one-hot targets
    Tensor Zt_onehot(Shape{{B, C}}, TensorOptions().with_req_grad(true));
    float* z_onehot_data = Zt_onehot.data<float>();
    for (int i = 0; i < B * C; ++i) {
        z_onehot_data[i] = z_data[i]; // Copy same logits
    }
    Value Z_onehot = make_tensor(Zt_onehot, "Logits_OneHot");
    
    Tensor Yt_onehot(Shape{{B, C}}, TensorOptions());
    float* y_onehot_data = Yt_onehot.data<float>();
    for (int i = 0; i < B * C; ++i) {
        y_onehot_data[i] = 0.0f;
    }
    y_onehot_data[0*C + 1] = 1.0f; // Row 0, class 1
    y_onehot_data[1*C + 2] = 1.0f; // Row 1, class 2
    y_onehot_data[2*C + 3] = 1.0f; // Row 2, class 3
    Value Y_onehot = make_tensor(Yt_onehot, "Targets_OneHot");
    
    // Compute one-hot CE
    Value loss_onehot = cross_entropy_with_logits(Z_onehot, Y_onehot);
    backward(loss_onehot);
    
    // Compare losses
    float loss_sparse_val = loss_sparse.val().data<float>()[0];
    float loss_onehot_val = loss_onehot.val().data<float>()[0];
    
    std::cout << "Sparse CE Loss: " << loss_sparse_val << "\n";
    std::cout << "One-Hot CE Loss: " << loss_onehot_val << "\n";
    
    if (std::abs(loss_sparse_val - loss_onehot_val) > 1e-4) {
        std::cerr << "Loss mismatch between sparse and one-hot!\n";
        return false;
    }
    
    // Compare gradients
    float* g_sparse = Z_sparse.grad().data<float>();
    float* g_onehot = Z_onehot.grad().data<float>();
    
    for (int i = 0; i < B * C; ++i) {
        if (std::abs(g_sparse[i] - g_onehot[i]) > 1e-4) {
            std::cerr << "Gradient mismatch at index " << i 
                      << ": sparse=" << g_sparse[i] << ", onehot=" << g_onehot[i] << "\n";
            return false;
        }
    }
    
    std::cout << "✓ Consistency test passed\n";
    return true;
}

int main() {
    std::cout << "=== Sparse Cross Entropy Edge Cases Test Suite ===\n";
    
    bool all_passed = true;
    
    all_passed &= test_single_class();
    all_passed &= test_many_classes();
    all_passed &= test_extreme_logits();
    all_passed &= test_large_batch();
    all_passed &= test_same_target_class();
    all_passed &= test_gradient_accumulation();
    all_passed &= test_zero_logits();
    all_passed &= test_int32_targets();
    all_passed &= test_gradient_magnitude();
    all_passed &= test_consistency_with_onehot();
    
    print_section("FINAL RESULTS");
    if (all_passed) {
        std::cout << "✓✓✓ ALL TESTS PASSED ✓✓✓\n";
        return 0;
    } else {
        std::cout << "✗✗✗ SOME TESTS FAILED ✗✗✗\n";
        return 1;
    }
}
