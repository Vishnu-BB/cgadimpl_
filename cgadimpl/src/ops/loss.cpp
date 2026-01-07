#include "ad/ops/nodeops.hpp"
#include <cuda_runtime.h>
#include "tensor.hpp" 
#include <unordered_map>
#include <cmath> 
#include <type_traits> 

namespace ag {
namespace detail {
std::shared_ptr<Node> mse_loss_nodeops(const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target) {
    Tensor diff = pred->value - target->value;
    Tensor sq   = diff * diff;
    Tensor loss = OwnTensor::reduce_mean(sq); 
    auto n = std::make_shared<Node>(loss, Op::MSELoss, (pred->requires_grad()), "mseloss");
    n->inputs = {pred, target};
    if (pred) pred->child_grad_count++;
    if (target) target->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> mae_loss_nodeops(const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target) {
    Tensor diff = pred->value - target->value;
    Tensor abs_diff = OwnTensor::abs(diff, ag::current_stream());
    Tensor loss = OwnTensor::reduce_mean(abs_diff);
    auto n = std::make_shared<Node>(loss, Op::MAELoss, (pred->requires_grad() || target->requires_grad()), "maeloss");
    n->inputs = {pred, target};
    if (pred) pred->child_grad_count++;
    if (target) target->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
std::shared_ptr<Node> cross_entropy_with_logits_nodeops(const std::shared_ptr<Node>& logits, const std::shared_ptr<Node>& onehot){
    const Tensor& Z = logits->value;
    const Tensor& Y = onehot->value;
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor z_shifted = Z - max_val;
    Tensor log_sum_exp = OwnTensor::log(OwnTensor::reduce_sum(OwnTensor::exp(z_shifted, ag::current_stream()), {-1}, true), ag::current_stream());
    Tensor log_sm = z_shifted - log_sum_exp;
    Tensor prod = Y * log_sm;
    Tensor sum_prod = OwnTensor::reduce_sum(prod, {-1}); 
    Tensor loss = OwnTensor::reduce_mean(sum_prod * -1.0f); 
    auto n = std::make_shared<Node>(loss, Op::CeWithLogits, (logits->requires_grad() || onehot->requires_grad()), "ce_with_logits");
    n->inputs = {logits, onehot};
    if (logits) logits->child_grad_count++;
    if (onehot) onehot->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> kldivergence_nodeops(const std::shared_ptr<Node>& logits, const std::shared_ptr<Node>& onehot){
    const Tensor& Z = logits->value;
    const Tensor& Y = onehot->value;
    Tensor log_Y = OwnTensor::log(Y + 1e-9f, ag::current_stream());
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor z_shifted = Z - max_val;
    Tensor log_sum_exp = OwnTensor::log(OwnTensor::reduce_sum(OwnTensor::exp(z_shifted, ag::current_stream()), {-1}, true), ag::current_stream());
    Tensor log_sm_Z = z_shifted - log_sum_exp;
    Tensor kl_div_elementwise = Y * (log_Y - log_sm_Z);
    Tensor sum_kl = OwnTensor::reduce_sum(kl_div_elementwise, {-1});
    Tensor loss = OwnTensor::reduce_mean(sum_kl);
    auto n = std::make_shared<Node>(loss, Op::KLDivergence, (logits->requires_grad() || onehot->requires_grad()), "kldivergence");
    n->inputs = {logits, onehot};
    if (logits) logits->child_grad_count++;
    if (onehot) onehot->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> sparse_cross_entropy_with_logits_nodeops(const std::shared_ptr<Node>& logits, const std::shared_ptr<Node>& target){
    const Tensor& Z = logits->value;
    const Tensor& Y = target->value;
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor z_shifted = Z - max_val;
    Tensor log_sum_exp = OwnTensor::log(OwnTensor::reduce_sum(OwnTensor::exp(z_shifted, ag::current_stream()), {-1}, true), ag::current_stream());
    Tensor log_sm_Z = z_shifted - log_sum_exp;

    // Gather logic: select log_prob[target] for each batch element
    // Note: This is a CPU implementation. For GPU, we would need a kernel.
    // Since we don't have a generic gather op exposed yet, we implement it here.
    
    Tensor selected_log_probs;
    if (Z.is_cpu() && Y.is_cpu()) {
        selected_log_probs = OwnTensor::Tensor::zeros(OwnTensor::Shape{{Z.shape().dims[0]}}, ag::options(Z));
        
        // Simple dispatch for float/double
        auto dtype = Z.dtype();
        if (dtype == Dtype::Float32) {
            const float* log_probs = log_sm_Z.data<float>();
            float* out = selected_log_probs.data<float>();
            // Handle target types (assume Int64 or Int32)
            if (Y.dtype() == Dtype::Int64) {
                const int64_t* targets = Y.data<int64_t>();
                int64_t batch_size = Z.shape().dims[0];
                int64_t num_classes = Z.shape().dims[1];
                for(int64_t i=0; i<batch_size; ++i) {
                    int64_t t = targets[i];
                    if (t >= 0 && t < num_classes) {
                        out[i] = log_probs[i * num_classes + t];
                    } else {
                        out[i] = 0.0f; // Ignore invalid indices? Or throw?
                    }
                }
            } else if (Y.dtype() == Dtype::Int32) {
                const int32_t* targets = Y.data<int32_t>();
                int64_t batch_size = Z.shape().dims[0];
                int64_t num_classes = Z.shape().dims[1];
                for(int64_t i=0; i<batch_size; ++i) {
                    int32_t t = targets[i];
                    if (t >= 0 && t < num_classes) {
                        out[i] = log_probs[i * num_classes + t];
                    } else {
                        out[i] = 0.0f;
                    }
                }
            } else {
                 throw std::runtime_error("SparseCE: Targets must be Int32 or Int64");
            }
        } else {
             throw std::runtime_error("SparseCE: Logits must be Float32 for now");
        }
    } else {
        // Fallback or Error for GPU
        // If we are on GPU, we can't easily iterate. 
        // We MUST rely on OwnTensor ops.
        // If OwnTensor doesn't have gather, we are stuck.
        // BUT, we can try to use one_hot + sum if we can't gather.
        // Since the user specifically asked for "gather logic", and we are on GPU,
        // we might be forced to use the one_hot trick if gather isn't available.
        // However, I will throw for now or try one_hot if available.
        // Let's assume we can use one_hot if it exists (which I'm not sure of).
        // For safety, I'll throw "Not implemented for GPU" and let the user know.
        throw std::runtime_error("SparseCE: GPU support requires 'gather' or 'one_hot' kernel in TensorLib");
    }

    Tensor loss = OwnTensor::reduce_mean(selected_log_probs * -1.0f); 
    auto n = std::make_shared<Node>(loss, Op::SparseCeWithLogits, (logits->requires_grad() || target->requires_grad()), "sparse_ce_with_logits");
    n->inputs = {logits, target};
    if (logits) logits->child_grad_count++;
    if (target) target->child_grad_count++;
    ag::debug::on_node_created(n);
    return n;
}
} // namespace detail
} // namespace ag