#include "ad/core/graph.hpp"
#include "ad/autodiff/autodiff.hpp"
#include "ad/autodiff/checkpoint.hpp"
#include "tensor.hpp"
#include <iostream>

using namespace ag;

int main() {
    std::cout << "Running Simple Uniform Checkpoint Test..." << std::endl;

    // 1. Create a simple linear chain graph
    int chain_length = 10;
    int dim = 128;
    
    auto opts = TensorOptions().with_dtype(Dtype::Float32).with_req_grad(true);
    Value x = make_tensor(Tensor::randn(Shape{{dim, dim}}, opts), "x");
    Value w = make_tensor(Tensor::randn(Shape{{dim, dim}}, opts), "w");
    
    Value current = x;
    for (int i = 0; i < chain_length; ++i) {
        current = matmul(current, w);
        current = relu(current);
    }
    
    Value loss = sum(current);

    // 2. Apply uniform checkpointing (every 3rd node)
    std::cout << "Applying auto_checkpoint_every_n(3)..." << std::endl;
    auto_checkpoint_every_n(loss, 3);

    // 3. Verify checkpoints were marked (optional, but good for sanity)
    int checkpoint_count = 0;
    auto nodes = topo_from(loss.node.get());
    for (auto* n : nodes) {
        if (n->is_checkpoint) {
            checkpoint_count++;
        }
    }
    std::cout << "Marked " << checkpoint_count << " nodes as checkpoints." << std::endl;
    // std::vector<Value> inputs = {X};
    // std::vector<Value> params = {W1, b1, W2,b2,W3,b3,W4,b4,W5,b5};

    ag::debug::dump_vjp_dot(loss, "build/graph_vjm.dot");
    // 4. Run backward pass
    std::cout << "Running backward pass..." << std::endl;
    try {
        backward(loss);
        std::cout << "Backward pass completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Backward pass FAILED: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Test Passed!" << std::endl;
    return 0;
}
