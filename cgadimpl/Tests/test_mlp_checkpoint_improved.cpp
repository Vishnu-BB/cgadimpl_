#include "ad/core/graph.hpp"
#include "ad/autodiff/autodiff.hpp"
#include "ad/autodiff/checkpoint.hpp"
#include "ad/autodiff/careful_deletion.hpp"
#include "ad/autodiff/inplace.hpp"
#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <unordered_set>

using namespace ag;

// ============================================================================
// Helper: Calculate Total Graph Memory
// ============================================================================

size_t calculate_graph_memory(const Value& root) {
    if (!root.node) return 0;
    
    auto nodes = topo_from(root.node.get());
    size_t total_bytes = 0;
    
    for (Node* n : nodes) {
        // Count value memory
        if (n->value.numel() > 0) {
            total_bytes += n->value.numel() * sizeof(float);
        }
        // Count gradient memory
        if (n->grad.numel() > 0) {
            total_bytes += n->grad.numel() * sizeof(float);
        }
    }
    
    return total_bytes;
}

// ============================================================================
// Model Definition
// ============================================================================

struct DeepModel {
    std::vector<Value> weights;
    std::vector<Value> biases;
    int depth;
    int hidden_dim;
    
    DeepModel(int d, int h) : depth(d), hidden_dim(h) {
        auto opts = TensorOptions().with_dtype(Dtype::Float32).with_req_grad(true); 
        for (int i = 0; i < depth; ++i) {
            weights.push_back(make_tensor(Tensor::randn(Shape{{h, h}}, opts), ("w" + std::to_string(i)).c_str()));
            biases.push_back(make_tensor(Tensor::randn(Shape{{1, h}}, opts), ("b" + std::to_string(i)).c_str()));
        }
    }
    
    Value forward(Value x, bool use_checkpointing) {
        for (int i = 0; i < depth; ++i) {
            // Linear layer
            x = matmul(x, weights[i]) + biases[i];
            x = relu(x);
            
            // Checkpoint every 3rd layer if enabled
            // This marks "anchors" - points we definitely want to save
            if (use_checkpointing && (i > 0) && (i % 3 == 0) && (i < depth - 1)) {
                checkpoint_impl::mark_node_checkpoint(x.node, CheckpointOptions());
            }
        }
        return x;
    }
};

// ============================================================================
// Test Runner
// ============================================================================

void run_improved_checkpoint_test() {
    std::cout << "==================================================\n";
    std::cout << "      Improved Gradient Checkpointing Test        \n";
    std::cout << "==================================================\n\n";
    
    int depth = 10;
    int hidden_dim = 1024;
    int batch_size = 64;
    
    std::cout << "Model Config:\n";
    std::cout << "  - Depth: " << depth << " layers\n";
    std::cout << "  - Hidden Dim: " << hidden_dim << "\n";
    std::cout << "  - Batch Size: " << batch_size << "\n\n";
    
    DeepModel model(depth, hidden_dim);
    Value input = make_tensor(Tensor::randn(Shape{{batch_size, hidden_dim}}, 
                          TensorOptions().with_dtype(Dtype::Float32)), "input");    
    
    // ------------------------------------------------------------------------
    // Scenario: True Gradient Checkpointing
    // ------------------------------------------------------------------------
    std::cout << "--- Running Forward Pass with Checkpoints ---\n";
    
    // Forward
    Value out = model.forward(input, true);
    
    size_t mem_initial = calculate_graph_memory(out);
    std::cout << "  Initial Memory (Activations): " << (mem_initial / 1024.0 / 1024.0) << " MB\n";
    
    // Identify Anchors
    std::unordered_set<Node*> anchors;
    auto nodes = topo_from(out.node.get());
    for (Node* n : nodes) {
        if (n->is_checkpoint) {
            anchors.insert(n);
        }
    }
    std::cout << "  Identified " << anchors.size() << " anchor checkpoints.\n";
    
    // Perform Cleanup using ForwardPass policy
    // This should delete intermediates even if they are NOT marked as checkpoints
    std::cout << "  Performing memory cleanup (ForwardPass policy)...\n";
    memory::sweep_safe_nodes(out, memory::DeletePolicy::ForwardPass, anchors);
    
    size_t mem_cleaned = calculate_graph_memory(out);
    std::cout << "  Memory After Cleanup: " << (mem_cleaned / 1024.0 / 1024.0) << " MB\n";
    
    if (mem_cleaned >= mem_initial) {
        std::cout << "❌ FAILURE: No memory savings observed. Cleanup failed.\n";
        throw std::runtime_error("Memory cleanup failed");
    }
    
    std::cout << "✅ SUCCESS: Memory reduced by " << ((mem_initial - mem_cleaned) / 1024.0 / 1024.0) << " MB\n";

    // Verify Backward
    std::cout << "\n--- Verifying Backward Pass (Recomputation) ---\n";
    try {
        Value loss = sum(out);
        backward(loss);
        std::cout << "  Backward pass completed successfully.\n";
        
        // Check if gradients are populated
        if (model.weights[0].node->grad.numel() == 0) {
             throw std::runtime_error("Gradients were not computed!");
        }
        std::cout << "✅ SUCCESS: Gradients computed.\n";
        
    } catch (const std::exception& e) {
        std::cout << "❌ Backward pass failed: " << e.what() << "\n";
        throw;
    }
}

int main() {
    try {
        run_improved_checkpoint_test();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
