#include "ad/ag_all.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

using namespace ag;
using namespace OwnTensor;

int main() {

    std::cout << "\nStarting raw tensor performance test (no graph)..." << std::endl;
    auto start_time_raw = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10000; ++i) {
        Tensor A = Tensor::randn(Shape{{100, 100}}, TensorOptions().with_req_grad(false));
        Tensor B = Tensor::randn(Shape{{100, 100}}, TensorOptions().with_req_grad(false));
        
        // Raw tensor multiplication
        Tensor C = matmul(A, B);

        // if (i % 100 == 0) {
        //      std::cout << "Iteration " << i << " result shape: " << C.shape().dims[0] << "," << C.shape().dims[1] << std::endl;
        //     //  std::cout << "C (raw):\n";
        //     //  C.display(std::cout);
        // }
    }

    auto end_time_raw = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_raw = end_time_raw - start_time_raw;

    std::cout << "Finished 10000 raw tensor operations." << std::endl;
    std::cout << "Total time taken (raw): " << elapsed_raw.count() << " seconds." << std::endl;
    std::cout << "Average time per operation (raw): " << elapsed_raw.count() / 1000.0 << " seconds." << std::endl;
    
    std::cout << "Starting performance test..." << std::endl;

    // Setup random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10000; ++i) {
        // Create random tensors
        // Using a small size for the loop to be reasonably fast but still do work
        Tensor A = Tensor::randn(Shape{{100, 100}}, TensorOptions().with_req_grad(true));
        Tensor B = Tensor::randn(Shape{{100, 100}}, TensorOptions().with_req_grad(true));

        auto va = ag::make_tensor(A, "A");
        auto vb = ag::make_tensor(B, "B");

        // Perform an operation (MatMul)
        auto result_value = ag::matmul(va, vb);
        
        // Force evaluation if lazy (though current impl seems eager or at least constructs the graph)
        // Accessing value to ensure computation happens if it's lazy-evaluated upon access
        // (Assuming standard eager or graph construction overhead is what we want to measure)
        
        // Print output values for the first few iterations to verify, or maybe just the last one to avoid spam
        // if (i % 100 == 0) {
        //      // std::cout << "c:\n";
        //      // C.display(std::cout);
        //      std::cout << "Iteration " << i << " result shape: " << result_value.shape()[0] << "," << result_value.shape()[1] << std::endl;
        //      // Printing the whole tensor might be too much, let's print a value
        //      // std::cout << result_node->value << std::endl; 
        // }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Finished 10000 operations." << std::endl;
    std::cout << "Total time taken: " << elapsed.count() << " seconds." << std::endl;
    std::cout << "Average time per operation: " << elapsed.count() / 1000.0 << " seconds." << std::endl;

    

    return 0;
}
