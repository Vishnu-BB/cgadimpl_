#include "ad/ag_all.hpp"
#include "ad/debug.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <functional>

using namespace ag;
using namespace OwnTensor;

void run_test_case(const std::string& name, std::function<void()> test_func) {
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Running Test Case: " << name << std::endl;
    try {
        test_func();
        std::cout << "[PASSED] " << name << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[FAILED] " << name << " - Exception: " << e.what() << std::endl;
    }
    std::cout << "--------------------------------------------------" << std::endl;
}

int main() {
    std::cout << "Starting Graph Edge Cases Test..." << std::endl;

    // 1. Disconnected Graph
    run_test_case("Disconnected Graph", []() {
        auto A = make_tensor(Tensor::randn(Shape{{2, 2}}, TensorOptions()), "A");
        auto B = make_tensor(Tensor::randn(Shape{{2, 2}}, TensorOptions()), "B");
        auto C = A + A; // Graph 1
        auto D = B * B; // Graph 2
        
        std::cout << "Graph 1 (A+A) and Graph 2 (B*B) created independently." << std::endl;
        ag::debug::dump_dot(C, "disconnected_graph_1.dot");
        ag::debug::dump_dot(D, "disconnected_graph_2.dot");
    });

    // 2. Diamond Graph
    run_test_case("Diamond Graph", []() {
        auto A = make_tensor(Tensor::randn(Shape{{2, 2}}, TensorOptions()), "A");
        auto B = A * 2.0f;
        auto C = A + 1.0f;
        auto D = B + C;
        
        std::cout << "Diamond structure created: A->B, A->C, B+C->D" << std::endl;
        ag::debug::dump_dot(D, "diamond_graph.dot");
    });

    // 3. Scalar Operations (Simulated with 1-element vector)
    run_test_case("Scalar Operations", []() {
        auto A = make_tensor(Tensor::randn(Shape{{1}}, TensorOptions()), "A_scalar"); // 1-element tensor
        auto B = make_tensor(Tensor::randn(Shape{{1}}, TensorOptions()), "B_scalar");
        auto C = A + B;
        
        std::cout << "Scalar addition result shape: " << C.shape()[0] << " dims" << std::endl;
        ag::debug::dump_dot(C, "scalar_graph.dot");
    });

    // 4. Broadcasting
    run_test_case("Broadcasting", []() {
        auto A = make_tensor(Tensor::randn(Shape{{2, 2}}, TensorOptions()), "A_2x2");
        auto B = make_tensor(Tensor::randn(Shape{{2}}, TensorOptions()), "B_2");
        auto C = A + B; // Implicit broadcast
        
        std::cout << "Broadcast addition (2x2 + 2) result shape: " << C.shape()[0] << "," << C.shape()[1] << std::endl;
        ag::debug::dump_dot(C, "broadcast_graph.dot");
    });

    // 5. Deep Graph
    run_test_case("Deep Graph", []() {
        auto A = make_tensor(Tensor::randn(Shape{{2, 2}}, TensorOptions()), "A");
        auto curr = A;
        for (int i = 0; i < 50; ++i) {
            curr = curr + 1.0f;
        }
        std::cout << "Deep graph with 50 additions created." << std::endl;
        ag::debug::dump_dot(curr, "deep_graph.dot");
    });

    // 6. Wide Graph
    run_test_case("Wide Graph", []() {
        std::vector<Value> inputs;
        for (int i = 0; i < 50; ++i) {
            std::string name = "In_" + std::to_string(i);
            inputs.push_back(make_tensor(Tensor::randn(Shape{{2, 2}}, TensorOptions()), name.c_str()));
        }
        
        auto sum = inputs[0];
        for (size_t i = 1; i < inputs.size(); ++i) {
            sum = sum + inputs[i];
        }
        std::cout << "Wide graph summing 50 inputs created." << std::endl;
        ag::debug::dump_dot(sum, "wide_graph.dot");
    });

    // 7. Reused Node
    run_test_case("Reused Node", []() {
        auto A = make_tensor(Tensor::randn(Shape{{2, 2}}, TensorOptions()), "A");
        auto B = A + A + A;
        
        std::cout << "Node A reused 3 times in sum." << std::endl;
        ag::debug::dump_dot(B, "reused_node_graph.dot");
    });

    // 8. Mixed Grads
    run_test_case("Mixed Grads", []() {
        auto A = make_tensor(Tensor::randn(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "A_grad");
        auto B = make_tensor(Tensor::randn(Shape{{2, 2}}, TensorOptions().with_req_grad(false)), "B_no_grad");
        auto C = A + B;
        
        std::cout << "Mixed grad addition (Grad + NoGrad). Result requires_grad: " << C.val().requires_grad() << std::endl;
        ag::debug::dump_dot(C, "mixed_grads_graph.dot");
    });

    // 9. Zero-sized Tensor (if supported)
    run_test_case("Zero-sized Tensor", []() {
        // Note: Check if 0-sized dimension is supported by TensorLib
        try {
            auto A = make_tensor(Tensor::randn(Shape{{0, 5}}, TensorOptions()), "A_empty");
            auto B = make_tensor(Tensor::randn(Shape{{0, 5}}, TensorOptions()), "B_empty");
            auto C = A + B;
            std::cout << "Zero-sized tensor addition result shape: " << C.shape()[0] << "," << C.shape()[1] << std::endl;
            ag::debug::dump_dot(C, "zero_sized_graph.dot");
        } catch (...) {
            std::cout << "Zero-sized tensors might not be fully supported or behaved differently." << std::endl;
        }
    });

    // 10. Unused Branch
    run_test_case("Unused Branch", []() {
        auto A = make_tensor(Tensor::randn(Shape{{2, 2}}, TensorOptions()), "A");
        auto B = make_tensor(Tensor::randn(Shape{{2, 2}}, TensorOptions()), "B");
        auto Unused = A * B; // Created but not used in Final
        auto Final = A + 1.0f;
        
        std::cout << "Graph with unused branch created." << std::endl;
        // Dumping Final should NOT show Unused if it's truly disconnected from Final's history
        ag::debug::dump_dot(Final, "unused_branch_graph.dot");
    });

    return 0;
}
