#include "ad/core/graph.hpp"
#include "ad/autodiff/autodiff.hpp"
#include "ad/autodiff/checkpoint.hpp"
#include "ad/autodiff/careful_deletion.hpp"
#include "ad/autodiff/inplace.hpp"
#include "tensor.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <iomanip>

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
// 4-Layer Model Definition: tanh -> relu -> sigmoid -> gelu
// ============================================================================

struct FourLayerModel {
    std::vector<Value> weights;
    std::vector<Value> biases;
    int hidden_dim;
    
    FourLayerModel(int h, bool save_to_file = false) : hidden_dim(h) {
        auto opts = TensorOptions().with_dtype(Dtype::Float32).with_req_grad(true);
        
        // Use fixed seed for reproducibility
        std::srand(42);
        
        // Create 4 layers
        for (int i = 0; i < 4; ++i) {
            weights.push_back(make_tensor(
                Tensor::randn(Shape{{h, h}}, opts), 
                ("w" + std::to_string(i)).c_str()
            ));
            biases.push_back(make_tensor(
                Tensor::randn(Shape{{1, h}}, opts), 
                ("b" + std::to_string(i)).c_str()
            ));
        }
        
        // Save weights and biases to files for libtorch comparison
        if (save_to_file) {
            for (int i = 0; i < 4; ++i) {
                std::string w_file = "weight_" + std::to_string(i) + ".bin";
                std::string b_file = "bias_" + std::to_string(i) + ".bin";
                
                // Save weight
                auto w_data = weights[i].node->value.data<float>();
                std::ofstream wf(w_file, std::ios::binary);
                wf.write(reinterpret_cast<const char*>(w_data), h * h * sizeof(float));
                wf.close();
                
                // Save bias
                auto b_data = biases[i].node->value.data<float>();
                std::ofstream bf(b_file, std::ios::binary);
                bf.write(reinterpret_cast<const char*>(b_data), h * sizeof(float));
                bf.close();
            }
            std::cout << "  Saved weights and biases to files for libtorch comparison.\n";
        }
    }
    
    Value forward(Value x, bool use_checkpointing) {
        // Layer 1: Linear + Tanh
        x = matmul(x, weights[0]) + biases[0];
        x = tanh(x);
        if (use_checkpointing) {
            checkpoint_impl::mark_node_checkpoint(x.node, CheckpointOptions());
        }
        
        // Layer 2: Linear + ReLU
        x = matmul(x, weights[1]) + biases[1];
        x = relu(x);
        if (use_checkpointing) {
            checkpoint_impl::mark_node_checkpoint(x.node, CheckpointOptions());
        }
        
        // Layer 3: Linear + Sigmoid
        x = matmul(x, weights[2]) + biases[2];
        x = sigmoid(x);
        if (use_checkpointing) {
            checkpoint_impl::mark_node_checkpoint(x.node, CheckpointOptions());
        }
        
        // Layer 4: Linear + GELU
        x = matmul(x, weights[3]) + biases[3];
        x = gelu(x);
        if (use_checkpointing) {
            checkpoint_impl::mark_node_checkpoint(x.node, CheckpointOptions());
        }
        
        return x;
    }
};

// ============================================================================
// Test Runner
// ============================================================================

void run_cgadimpl_comparison() {
    std::cout << "==================================================\n";
    std::cout << "      cgadimpl Checkpoint Memory Comparison      \n";
    std::cout << "==================================================\n\n";
    
    int hidden_dim = 1024;
    int batch_size = 128;
    
    std::cout << "Model Config:\n";
    std::cout << "  - Layers: 4 (tanh, relu, sigmoid, gelu)\n";
    std::cout << "  - Hidden Dim: " << hidden_dim << "\n";
    std::cout << "  - Batch Size: " << batch_size << "\n";
    std::cout << "  - Random Seed: 42 (for reproducibility)\n\n";
    
    FourLayerModel model(hidden_dim, true);  // Save weights to files
    
    // Create input with fixed seed
    std::srand(42);
    Value input = make_tensor(
        Tensor::randn(Shape{{batch_size, hidden_dim}}, 
                      TensorOptions().with_dtype(Dtype::Float32)), 
        "input"
    );
    
    // Save input to file for libtorch comparison
    auto input_data = input.node->value.data<float>();
    std::ofstream input_file("input.bin", std::ios::binary);
    input_file.write(reinterpret_cast<const char*>(input_data), 
                     batch_size * hidden_dim * sizeof(float));
    input_file.close();
    std::cout << "  Saved input data to input.bin for libtorch comparison.\n\n";
    
    // ------------------------------------------------------------------------
    // Scenario 1: No Checkpointing
    // ------------------------------------------------------------------------
    std::cout << "--- Scenario 1: No Checkpointing ---\n";
    
    Value out_no_cp = model.forward(input, false);
    size_t mem_forward_no_cp = calculate_graph_memory(out_no_cp);
    std::cout << "  Peak Memory (Forward): " 
              << std::fixed << std::setprecision(2)
              << (mem_forward_no_cp / 1024.0 / 1024.0) << " MB\n";
    
    // Backward pass
    Value loss_no_cp = sum(out_no_cp);
    backward(loss_no_cp);
    
    size_t mem_after_backward_no_cp = calculate_graph_memory(out_no_cp);
    std::cout << "  Memory After Backward: " 
              << (mem_after_backward_no_cp / 1024.0 / 1024.0) << " MB\n";
    
    // Cleanup for next scenario
    out_no_cp = Value();
    loss_no_cp = Value();
    
    // ------------------------------------------------------------------------
    // Scenario 2: With Checkpointing
    // ------------------------------------------------------------------------
    std::cout << "\n--- Scenario 2: With Checkpointing ---\n";
    
    Value out_cp = model.forward(input, true);
    size_t mem_forward_cp = calculate_graph_memory(out_cp);
    std::cout << "  Peak Memory (Forward): " 
              << (mem_forward_cp / 1024.0 / 1024.0) << " MB\n";
    
    // Identify anchor checkpoints
    std::unordered_set<Node*> anchors;
    auto nodes = topo_from(out_cp.node.get());
    for (Node* n : nodes) {
        if (n->is_checkpoint) {
            anchors.insert(n);
        }
    }
    std::cout << "  Identified " << anchors.size() << " anchor checkpoints.\n";
    
    // Perform memory cleanup
    std::cout << "  Performing memory cleanup...\n";
    memory::sweep_safe_nodes(out_cp, memory::DeletePolicy::ForwardPass, anchors);
    
    size_t mem_after_cleanup = calculate_graph_memory(out_cp);
    std::cout << "  Memory After Cleanup: " 
              << (mem_after_cleanup / 1024.0 / 1024.0) << " MB\n";
    
    // Backward pass with recomputation
    std::cout << "  Running backward pass (with recomputation)...\n";
    Value loss_cp = sum(out_cp);
    backward(loss_cp);
    
    size_t mem_after_backward_cp = calculate_graph_memory(out_cp);
    std::cout << "  Memory After Backward: " 
              << (mem_after_backward_cp / 1024.0 / 1024.0) << " MB\n";
    
    // Print checkpoint statistics
    print_checkpoint_stats();
    
    // ------------------------------------------------------------------------
    // Results Summary
    // ------------------------------------------------------------------------
    std::cout << "\n--------------------------------------------------\n";
    std::cout << "Results Summary:\n";
    std::cout << "  No Checkpointing:\n";
    std::cout << "    Forward Memory:  " 
              << (mem_forward_no_cp / 1024.0 / 1024.0) << " MB\n";
    std::cout << "    Backward Memory: " 
              << (mem_after_backward_no_cp / 1024.0 / 1024.0) << " MB\n";
    
    std::cout << "\n  With Checkpointing:\n";
    std::cout << "    Forward Memory:  " 
              << (mem_forward_cp / 1024.0 / 1024.0) << " MB\n";
    std::cout << "    Cleanup Memory:  " 
              << (mem_after_cleanup / 1024.0 / 1024.0) << " MB\n";
    std::cout << "    Backward Memory: " 
              << (mem_after_backward_cp / 1024.0 / 1024.0) << " MB\n";
    
    size_t saved = mem_forward_no_cp - mem_after_cleanup;
    double percent = 100.0 * saved / mem_forward_no_cp;
    
    std::cout << "\n  Memory Saved: " 
              << (saved / 1024.0 / 1024.0) << " MB (" 
              << std::setprecision(1) << percent << "%)\n";
    std::cout << "--------------------------------------------------\n";
    
    if (saved > 0) {
        std::cout << "\n✅ SUCCESS: Checkpointing reduced memory usage!\n";
    } else {
        std::cout << "\n❌ FAILURE: No memory savings observed.\n";
    }
}

int main() {
    try {
        run_cgadimpl_comparison();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
