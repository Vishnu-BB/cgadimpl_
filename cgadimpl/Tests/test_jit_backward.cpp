#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include "ad/runtime/jit_compiler.hpp"

using namespace ag;

int main() {
    std::cout << "===== JIT BACKWARD TEST =====\n";

    // ---------- Shapes & Data ----------
    const int B = 4;
    const int In = 8;
    const int Out = 4;

    auto opts_const = TensorOptions();
    Tensor Xt = Tensor::randn(Shape{{B, In}}, opts_const);
    Value X = make_tensor(Xt, "X");

    auto opts_param = TensorOptions().with_req_grad(true);
    auto W = make_tensor(Tensor::randn(Shape{{In, Out}}, opts_param), "W");
    auto b = make_tensor(Tensor::zeros(Shape{{1, Out}}, opts_param), "b");

    // ---------- Forward Pass ----------
    Value Y = matmul(X, W) + b;
    Value loss = sum(Y);

    // ---------- Eager Backward ----------
    loss.backward();
    Tensor eager_loss = loss.val();
    Tensor eager_grad_W = W.grad();
    Tensor eager_grad_b = b.grad();

    std::cout << "Eager pass completed.\n";

    // ---------- JIT Compilation ----------
    std::cout << "\nCompiling combined graph (forward + backward)...\n";

    std::vector<Value> inputs = {X};
    std::vector<Value> params = {W, b};

    auto comp = ag::jit::compile_with_backward(loss, inputs, params);

    std::cout << "Graph compilation successful.\n";

    // ---------- JIT Execution ----------
    std::cout << "\nRunning compiled graph...\n";

    std::vector<Tensor*> in_ptrs = {&X.node->value};
    std::vector<Tensor*> par_ptrs = {&W.node->value, &b.node->value};

    std::vector<Tensor> compiled_outputs;
    bool ok = comp.run(in_ptrs, par_ptrs, compiled_outputs);
    
    if (!ok) {
        std::cerr << "FAIL: JIT execution failed.\n";
        return 1;
    }

    if (compiled_outputs.size() != 3) {
        std::cerr << "FAIL: Expected 3 outputs (loss, grad_W, grad_b), got " << compiled_outputs.size() << ".\n";
        return 1;
    }

    Tensor compiled_loss = compiled_outputs[0];
    Tensor compiled_grad_W = compiled_outputs[1];
    Tensor compiled_grad_b = compiled_outputs[2];

    std::cout << "Compiled execution successful.\n";

    // ---------- Verification ----------
    auto compare = [](const std::string& name, const Tensor& eager, const Tensor& compiled) {
        float e_val = eager.to_cpu().data<float>()[0];
        float c_val = compiled.to_cpu().data<float>()[0];
        float diff = std::abs(e_val - c_val);
        std::cout << name << " - Eager: " << e_val << ", Compiled: " << c_val << ", Diff: " << diff << "\n";
        return diff < 1e-4f;
    };

    bool pass = true;
    pass &= compare("Loss", eager_loss, compiled_loss);
    pass &= compare("grad_W[0]", eager_grad_W, compiled_grad_W);
    pass &= compare("grad_b[0]", eager_grad_b, compiled_grad_b);

    if (pass) {
        std::cout << "✅ PASS: Eager and compiled results match.\n";
    } else {
        std::cout << "❌ FAIL: Results do not match.\n";
    }

    return pass ? 0 : 1;
}
