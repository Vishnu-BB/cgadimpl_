#include "ad/ops/nodeops.hpp"
#include <cuda_runtime.h>
#include "tensor.hpp" 
#include <unordered_map>
#include <cmath> 
#include <type_traits> 

namespace ag {
namespace detail {


std::shared_ptr<Node> matmul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
    Tensor A = a->value;
    Tensor B = b->value;
    if (A.dtype() != B.dtype()) {
        if (A.dtype() == Dtype::Bfloat16) A = A.as_type(Dtype::Float32);
        if (B.dtype() == Dtype::Bfloat16) B = B.as_type(Dtype::Float32);
    }
    Tensor C = matmul(A, B);
    auto n = std::make_shared<Node>(C, Op::MatMul, (a->requires_grad() || b->requires_grad()), "matmul");
    n->inputs = {a, b};
    if(a) a->child_grad_count++;
    if(b) b->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> linear_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c) {
    Tensor input_X = a->value;
    Tensor weight_W = b->value; 
    Tensor bias_b = c->value;

    // Promote for matmul
    if (input_X.dtype() != weight_W.dtype()) {
        if (input_X.dtype() == Dtype::Bfloat16) input_X = input_X.as_type(Dtype::Float32);
        if (weight_W.dtype() == Dtype::Bfloat16) weight_W = weight_W.as_type(Dtype::Float32);
    }
    
    // Matmul result will be F32 if promoted. Bias might need promotion.
    Tensor y_matmul = matmul(input_X, weight_W.t());
    
    if (y_matmul.dtype() != bias_b.dtype()) {
        if (bias_b.dtype() == Dtype::Bfloat16) bias_b = bias_b.as_type(Dtype::Float32);
        // If y_matmul is BF16 (unlikely if we promoted above) and bias is F32, promote y_matmul?
        // But we assume F32 dominance.
    }

    Tensor y = y_matmul + bias_b;
    auto n = std::make_shared<Node>(y, Op::Linear, (a->requires_grad() || b->requires_grad() || c->requires_grad()), "linear");
    n->inputs = {a, b, c};
    if (a) a->child_grad_count++;
    if (b) b->child_grad_count++;
    if (c) c->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> fmab_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c){
    Tensor A = a->value;
    Tensor B = b->value;
    Tensor C_val = c->value;

    if (A.dtype() != B.dtype()) {
        if (A.dtype() == Dtype::Bfloat16) A = A.as_type(Dtype::Float32);
        if (B.dtype() == Dtype::Bfloat16) B = B.as_type(Dtype::Float32);
    }
    
    Tensor y_matmul = matmul(A, B);
    
    if (y_matmul.dtype() != C_val.dtype()) {
        if (C_val.dtype() == Dtype::Bfloat16) C_val = C_val.as_type(Dtype::Float32);
    }

    Tensor y = y_matmul + C_val;
    auto n = std::make_shared<Node>(y, Op::FMA, (a->requires_grad() || b->requires_grad() || c->requires_grad()), "fmab");
    n->inputs = {a, b, c};
    if(a) a->child_grad_count++;
    if(b) b->child_grad_count++;
    if(c) c->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> transpose_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = x->value.t();
    auto n = std::make_shared<Node>(y, Op::Transpose, x->requires_grad(), "transpose");
    n->inputs = {x};
    if(x) x->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

} // namespace detail
} // namespace ag