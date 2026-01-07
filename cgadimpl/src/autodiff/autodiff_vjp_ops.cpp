// ====================================================================
// FILE: cgadimpl/src/autodiff/autodiff_vjp_ops.cpp (GPU-Aware Version)
// ====================================================================

#include "ad/detail/autodiff_ops.hpp"
#include "ad/runtime/cuda_graphs.hpp"
#include <cmath>
#include <stdexcept> 
#include <iostream>
#include <iostream>
namespace ag {
namespace detail{

Tensor to_fp32_if_float(const Tensor& t) { 
    if (OwnTensor::is_float(t.dtype()) && t.dtype() != Dtype::Float32 && t.dtype() != Dtype::Float64) {
        return t.as_type(Dtype::Float32);
    }
    return t;
}

static Tensor reduce_for_broadcast(const Tensor& grad_in, const Tensor& target_val) {
    if (grad_in.shape() == target_val.shape()) {
        return grad_in;
    }
    const auto& grad_dims = grad_in.shape().dims;
    const auto& target_dims = target_val.shape().dims;
    std::vector<int64_t> axes_to_sum;
    int grad_ndim = grad_dims.size();
    int target_ndim = target_dims.size();

    for (int i = 0; i < grad_ndim; ++i) {
        int target_idx = i - (grad_ndim - target_ndim);
        if (target_idx < 0) {
            axes_to_sum.push_back(i);
        } else if (target_dims[target_idx] == 1 && grad_dims[i] > 1) {
            axes_to_sum.push_back(i);
        }
    }

    Tensor summed_grad = grad_in;
    if (!axes_to_sum.empty()) {
        summed_grad = OwnTensor::reduce_sum(grad_in, axes_to_sum, true);
    }

    if (summed_grad.shape() != target_val.shape()) {
        if (summed_grad.numel() == target_val.numel()) {
            return summed_grad.reshape(target_val.shape());
        } else {
            // Descriptive error for debugging
            std::string msg = "reduce_for_broadcast: shape mismatch. grad_in: ";
            for(auto d : grad_dims) msg += std::to_string(d) + ",";
            msg += " target: ";
            for(auto d : target_dims) msg += std::to_string(d) + ",";
            msg += " summed: ";
            for(auto d : summed_grad.shape().dims) msg += std::to_string(d) + ",";
            throw std::runtime_error(msg);
        }
    }
    return summed_grad;
}
// --- Basic Arithmetic ---
void vjp_Add(const VjpContext& ctx){
    Node* A = ctx.node->inputs[0].get(); 
    Node* B = ctx.node->inputs[1].get();
    if (A->requires_grad()) {
        if (A->value.numel() == 0) {  // empty tensor 
            std::cerr << "vjp_Add: A->value is empty! A=" << A << "\n";
        }
        A->grad += reduce_for_broadcast(ctx.gy, A->value);  // accumulate gradient
    }
    if (B->requires_grad()) {
        if (B->value.numel() == 0) {
            std::cerr << "vjp_Add: B->value is empty! B=" << B << "\n";
        }
        B->grad += reduce_for_broadcast(ctx.gy, B->value);
    }
}

void vjp_Sub(const VjpContext& ctx){
    Node* A = ctx.node->inputs[0].get(); 
    Node* B = ctx.node->inputs[1].get();
    if (A->requires_grad()) A->grad += reduce_for_broadcast(ctx.gy, A->value);
    if (B->requires_grad()) B->grad -= reduce_for_broadcast(ctx.gy, B->value);
}

void vjp_Mul(const VjpContext& ctx){
    Node* A = ctx.node->inputs[0].get(); 
    Node* B = ctx.node->inputs[1].get();
    if (A->requires_grad()) A->grad += reduce_for_broadcast(ctx.gy * ctx.input(1), A->value);
    if (B->requires_grad()) B->grad += reduce_for_broadcast(ctx.gy * ctx.input(0), B->value);
}

void vjp_Div(const VjpContext& ctx){
    Node* A_node = ctx.node->inputs[0].get();
    Node* B_node = ctx.node->inputs[1].get();
    Tensor A = ctx.input(0);
    Tensor B = ctx.input(1);
    if (A_node->requires_grad()) {
        A_node->grad += reduce_for_broadcast(ctx.gy / B, A_node->value);
    }
    if (B_node->requires_grad()) {
        B_node->grad -= reduce_for_broadcast(ctx.gy * A / (B * B), B_node->value);
    }
}

// Classic Activations ---------------
void vjp_Relu(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = ctx.input(0);
        Tensor mask = OwnTensor::Tensor::zeros(x.shape(), ag::options(x));
        if (x.is_cpu()) {
            dispatch_by_dtype(x.dtype(), [&](auto dummy){
                using T = decltype(dummy);
                const T* x_ptr = x.data<T>();
                T* m_ptr = mask.data<T>();
                for(int64_t i=0; i<x.numel(); ++i) {
                    if constexpr (std::is_same_v<T, OwnTensor::complex32_t> || 
                                  std::is_same_v<T, OwnTensor::complex64_t> || 
                                  std::is_same_v<T, OwnTensor::complex128_t>) {
                        if (x_ptr[i].real() > 0) m_ptr[i] = T(1.0f);
                    } else {
                        if (x_ptr[i] > T(0)) m_ptr[i] = T(1.0f);
                    }
                }
            });
        }
        X->grad += ctx.gy * mask;
    }
}

void vjp_Sigmoid(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor y = detail::to_fp32_if_float(ctx.node->value);
        X->grad += ctx.gy * y * (1.0f - y);
    }
}
void vjp_Tanh(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor y = detail::to_fp32_if_float(ctx.node->value);
        X->grad += ctx.gy * (1.0f - y * y);
    }
}
void vjp_Softplus(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = ctx.input(0);
        Tensor sig = 1.0f / (1.0f + OwnTensor::exp(x * -1.0f, ag::current_stream()));
        X->grad += ctx.gy * sig;
    }
}

// Smooth Activations (better gradient flow) ---
void vjp_GELU(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        const float c1 = 0.7978845608f; 
        const float c2 = 0.044715f;
        Tensor x = ctx.input(0);
        Tensor x3 = x * x * x;
        Tensor u = c1 * (x + c2 * x3);
        Tensor tu = OwnTensor::tanh(u);
        Tensor d_tu = 1.0f - tu * tu;
        Tensor du_dx = c1 * (1.0f + 3.0f * c2 * x * x);
        X->grad += ctx.gy * (0.5f * (1.0f + tu) + 0.5f * x * d_tu * du_dx);
    }
}

void vjp_SiLU(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = ctx.input(0);
        Tensor sig = 1.0f / (1.0f + OwnTensor::exp(x * -1.0f, ag::current_stream()));
        X->grad += ctx.gy * (sig + x * sig * (1.0f - sig));
    }
}

void vjp_Mish(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = ctx.input(0);
        Tensor sp = OwnTensor::log(1.0f + OwnTensor::exp(x, ag::current_stream()), ag::current_stream());
        Tensor tsp = OwnTensor::tanh(sp);
        Tensor sig = 1.0f / (1.0f + OwnTensor::exp(x * -1.0f, ag::current_stream()));
        X->grad += ctx.gy * (tsp + x * (1.0f - tsp * tsp) * sig);
    }
}

void vjp_LeakyRelu(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        float alpha = ctx.input(1).to_cpu().data<float>()[0];
        Tensor x = ctx.input(0);
        Tensor mask = OwnTensor::Tensor::zeros(x.shape(), ag::options(x));
        if (x.is_cpu()) {
            dispatch_by_dtype(x.dtype(), [&](auto dummy){
                using T = decltype(dummy);
                const T* x_ptr = x.data<T>();
                T* m_ptr = mask.data<T>();
                for(int64_t i=0; i<x.numel(); ++i) {
                    if constexpr (std::is_same_v<T, OwnTensor::complex32_t> || 
                                  std::is_same_v<T, OwnTensor::complex64_t> || 
                                  std::is_same_v<T, OwnTensor::complex128_t>) {
                        m_ptr[i] = (x_ptr[i].real() > 0) ? T(1.0f) : T(alpha);
                    } else {
                        m_ptr[i] = (x_ptr[i] > T(0)) ? T(1.0f) : T(alpha);
                    }
                }
            });
        }
        X->grad += ctx.gy * mask;
    }
}

// Specialized activations -----------
void vjp_Gaus(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor y = detail::to_fp32_if_float(ctx.node->value);
        X->grad += ctx.gy * y * (ctx.input(0) * -2.0f);
    }
}

void vjp_Parcon(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        X->grad += ctx.gy * (2.0f - ctx.input(0) * 2.0f);
    }
}

void vjp_LiSHT(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = ctx.input(0);
        Tensor tx = OwnTensor::tanh(x);
        X->grad += ctx.gy * (tx + x * (1.0f - tx * tx));
    }
}
// Standard Attention ---------
void vjp_Attention(const VjpContext& ctx){
    throw std::runtime_error("VJP for Attention not implemented yet!");
}

void vjp_SWIGLU(const VjpContext& ctx){
    throw std::runtime_error("VJP for SWIGLU not implemented yet!");
}

//Leaf -----------
void vjp_Leaf(const VjpContext&){ /* no-op */ }

//Unary Mathematical Functions ------------------
void vjp_Exp(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor y = detail::to_fp32_if_float(ctx.node->value);
        X->grad += ctx.gy * y;
    }
}

void vjp_Log(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) X->grad += ctx.gy / ctx.input(0);
}

void vjp_Sqrt(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor y = detail::to_fp32_if_float(ctx.node->value);
        X->grad += ctx.gy * 0.5f / y;
    }
}

void vjp_Reciprocal(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = ctx.input(0);
        X->grad -= ctx.gy / (x * x);
    }
}

void vjp_Sign(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) X->grad += ctx.gy * 0.0f;
}
void vjp_Abs(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        const float epsilon = 1e-9f;
        Tensor y = detail::to_fp32_if_float(ctx.node->value);
        X->grad += ctx.gy * (ctx.input(0) / y + epsilon);
    }
}
void vjp_Pow(const VjpContext& ctx){
    Node* A_node = ctx.node->inputs[0].get();
    Node* B_node = ctx.node->inputs[1].get();
    Tensor A = ctx.input(0);
    Tensor B = ctx.input(1);
    if (A_node->requires_grad()) {
        A_node->grad += reduce_for_broadcast(ctx.gy * B * OwnTensor::exp((B - 1.0f) * OwnTensor::log(A, ag::current_stream()), ag::current_stream()), A_node->value);
    }
    if (B_node->requires_grad()) {
        B_node->grad += reduce_for_broadcast(ctx.gy * detail::to_fp32_if_float(ctx.node->value) * OwnTensor::log(A, ag::current_stream()), B_node->value);
    }
}

//Core Matrix Operations ------------------------
void vjp_MatMul(const VjpContext& ctx){
    Node* A = ctx.node->inputs[0].get();
    Node* B = ctx.node->inputs[1].get();
    if (A->requires_grad()) A->grad += reduce_for_broadcast(OwnTensor::matmul(ctx.gy, ctx.input(1).t()), A->value);
    if (B->requires_grad()) B->grad += reduce_for_broadcast(OwnTensor::matmul(ctx.input(0).t(), ctx.gy), B->value);
}
// void vjp_MatMul(Node* n, const Tensor& gy) {
//     Node* A = n->inputs[0].get();
//     Node* B = n->inputs[1].get();

//     if (A->requires_grad()) {
//         Tensor grad_A = OwnTensor::matmul(gy, B->value.t());
//         // SUM GRADIENT IF A WAS BROADCASTED
//         if (grad_A.shape() != A->value.shape()) {
//             A->grad += sum_to_shape(grad_A, A->value.shape());
//         } else {
//             A->grad += grad_A;
//         }
//     }

//     if (B->requires_grad()) {
//         Tensor grad_B = OwnTensor::matmul(A->value.t(), gy);
//         // SUM GRADIENT IF B WAS BROADCASTED
//         if (grad_B.shape() != B->value.shape()) {
//             B->grad += sum_to_shape(grad_B, B->value.shape());
//         } else {
//             B->grad += grad_B;
//         }
//     }
// }
void vjp_Transpose(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) X->grad += ctx.gy.t();
}

//Fused Operations (better performance, fewer memory accesses) -------------

void vjp_Linear(const VjpContext& ctx){
    Node* X_node = ctx.node->inputs[0].get();
    Node* W_node = ctx.node->inputs[1].get();
    Node* b_node = ctx.node->inputs[2].get();
    if (X_node->requires_grad()) X_node->grad += OwnTensor::matmul(ctx.gy, ctx.input(1));
    if (W_node->requires_grad()) W_node->grad += OwnTensor::matmul(ctx.gy.t(), ctx.input(0));
    if (b_node->requires_grad()) b_node->grad += reduce_for_broadcast(ctx.gy, b_node->value);
}

void vjp_FMA(const VjpContext& ctx){
    Node* A = ctx.node->inputs[0].get();
    Node* B = ctx.node->inputs[1].get();
    Node* C = ctx.node->inputs[2].get();
    if (A->requires_grad()) A->grad += OwnTensor::matmul(ctx.gy, ctx.input(1).t());
    if (B->requires_grad()) B->grad += OwnTensor::matmul(ctx.input(0).t(), ctx.gy);
    if (C->requires_grad()) C->grad += ctx.gy;
}

//Classification losses ---------------
void vjp_CeWithLogits(const VjpContext& ctx){
    Node* Z_node = ctx.node->inputs[0].get();
    Tensor Z = ctx.input(0);
    Tensor Y = ctx.input(1);
    const float inv_batch_size = 1.0f / static_cast<float>(Z.shape().dims[0]);
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor z_shifted = Z - max_val;
    Tensor exp_z = OwnTensor::exp(z_shifted, ag::current_stream());
    Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
    Tensor softmax_z = exp_z / sum_exp_z;
    if (Z_node->requires_grad()) {
        float gy_val = ctx.gy.to_cpu().data<float>()[0];
        Z_node->grad += (softmax_z - Y) * (gy_val * inv_batch_size);
    }
}

void vjp_KLDivergence(const VjpContext& ctx){
    Node* Z_node = ctx.node->inputs[0].get();
    Node* Y_node = ctx.node->inputs[1].get();
    Tensor Z = ctx.input(0);
    Tensor Y = ctx.input(1);
    const float inv_batch_size = 1.0f / static_cast<float>(Z.shape().dims[0]);
    Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor z_shifted = Z - max_val;
    Tensor exp_z = OwnTensor::exp(z_shifted, ag::current_stream());
    Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
    Tensor softmax_z = exp_z / sum_exp_z;
    float gy_val = ctx.gy.to_cpu().data<float>()[0];
    if (Z_node->requires_grad()) {
        Z_node->grad += (softmax_z - Y) * (gy_val * inv_batch_size);
    }
    if (Y_node->requires_grad()) {
        Tensor log_Y = OwnTensor::log(Y + 1e-9f, ag::current_stream());
        Tensor log_softmax_z = z_shifted - OwnTensor::log(sum_exp_z, ag::current_stream());
        Y_node->grad += (log_Y + 1.0f - log_softmax_z) * (gy_val * inv_batch_size);
    }
}

void vjp_SparseCeWithLogits(const VjpContext& ctx){
    Node* Z_node = ctx.node->inputs[0].get();
    // We do not compute gradients for the integer target indices (inputs[1])

    if (Z_node->requires_grad()) {
        Tensor Z = ctx.input(0);      // Logits [Batch, NumClasses]
        Tensor Target = ctx.input(1); // Indices [Batch]

        // 1. Recompute Softmax (Identical to standard CE)
        Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
        Tensor z_shifted = Z - max_val;
        Tensor exp_z = OwnTensor::exp(z_shifted, ag::current_stream());
        Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
        Tensor softmax_z = exp_z / sum_exp_z;

        // 3. Compute Gradient
        // Scale by 1/BatchSize and incoming gradient (gy)
        float gy_val = ctx.gy.to_cpu().data<float>()[0];
        const float inv_batch_size = 1.0f / static_cast<float>(Z.shape().dims[0]);
        float scale = gy_val * inv_batch_size;
        
        // grad = (softmax - one_hot) * scale
        //      = softmax * scale - one_hot * scale
        
        Z_node->grad += softmax_z * scale;
        
        // Subtract scale from target indices
        if (Z.is_cpu() && Target.is_cpu()) {
             dispatch_by_dtype(Z.dtype(), [&](auto dummy){
                 using T = decltype(dummy);
                 // We need to modify Z_node->grad in place.
                 // Z_node->grad is a Tensor.
                 // Note: Z_node->grad might be a view or shared.
                 // But here we just added to it, so it should be valid.
                 
                 T* grad_ptr = Z_node->grad.data<T>();
                 int64_t batch_size = Z.shape().dims[0];
                 int64_t num_classes = Z.shape().dims[1];
                 
                 // Assume Int64 targets for now
                 if (Target.dtype() == Dtype::Int64) {
                     const int64_t* target_ptr = Target.data<int64_t>();
                     for(int64_t i=0; i<batch_size; ++i) {
                         int64_t t = target_ptr[i];
                         if (t >= 0 && t < num_classes) {
                             grad_ptr[i * num_classes + t] -= static_cast<T>(scale);
                         }
                     }
                 } else if (Target.dtype() == Dtype::Int32) {
                     const int32_t* target_ptr = Target.data<int32_t>();
                     for(int64_t i=0; i<batch_size; ++i) {
                         int32_t t = target_ptr[i];
                         if (t >= 0 && t < num_classes) {
                             grad_ptr[i * num_classes + t] -= static_cast<T>(scale);
                         }
                     }
                 } else {
                     throw std::runtime_error("SparseCE VJP: Targets must be Int32 or Int64");
                 }
             });
        } else {
             throw std::runtime_error("SparseCE VJP: GPU not supported yet (need kernel)");
        }
    }
}

//Regression Losses --------------
void vjp_MSELoss(const VjpContext& ctx){
    Node* Z_node = ctx.node->inputs[0].get();
    Node* Y_node = ctx.node->inputs[1].get();
    float gy_scalar = ctx.gy.to_cpu().data<float>()[0];
    const float scale = 2.0f / static_cast<float>(Z_node->value.numel());
    Tensor diff = ctx.input(0) - ctx.input(1);
    if (Z_node->requires_grad()) Z_node->grad += diff * (gy_scalar * scale);
    if (Y_node->requires_grad()) Y_node->grad -= diff * (gy_scalar * scale);
}

void vjp_MAELoss(const VjpContext& ctx){
    Node* Z_node = ctx.node->inputs[0].get();
    Node* Y_node = ctx.node->inputs[1].get();
    const float inv_N = 1.0f / static_cast<float>(Z_node->value.numel());
    const float epsilon = 1e-9f;
    Tensor diff = ctx.input(0) - ctx.input(1);
    Tensor sign_diff = diff / (OwnTensor::abs(diff, ag::current_stream()) + epsilon);
    float gy_val = ctx.gy.to_cpu().data<float>()[0];
    if (Z_node->requires_grad()) Z_node->grad += sign_diff * (gy_val * inv_N);
    if (Y_node->requires_grad()) Y_node->grad -= sign_diff * (gy_val * inv_N);
}

//Layer Normalization ------------
void vjp_LayerNorm(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = ctx.input(0);
        Tensor var = detail::to_fp32_if_float(*ctx.node->tape[0]);
        Tensor mean = detail::to_fp32_if_float(*ctx.node->tape[1]);
        Tensor std_inv = 1.0f / OwnTensor::sqrt(var + 1e-5f, ag::current_stream());
        int64_t D = x.shape().dims.back();
        Tensor d_xhat = ctx.gy * std_inv;
        Tensor d_var = OwnTensor::reduce_sum(ctx.gy * (x - mean) * -0.5f * (std_inv * std_inv * std_inv), {-1}, true);
        Tensor d_mean = OwnTensor::reduce_sum(ctx.gy * (std_inv * -1.0f), {-1}, true) + d_var * OwnTensor::reduce_sum((x - mean) * -2.0f, {-1}, true) / static_cast<float>(D);
        X->grad += d_xhat + d_var * (x - mean) * (2.0f / static_cast<float>(D)) + d_mean / static_cast<float>(D);
    }
}

//RMS Normalization -------------
void vjp_RMSNorm(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = ctx.input(0);
        Tensor rsqrt_var = detail::to_fp32_if_float(*ctx.node->tape[0]);
        int64_t D = x.shape().dims.back();
        Tensor dot = OwnTensor::reduce_sum(ctx.gy * x, {-1}, true);
        X->grad += rsqrt_var * (ctx.gy - x * (rsqrt_var * rsqrt_var) * dot / static_cast<float>(D));
    }
}

void vjp_RealRMSNorm(const VjpContext& ctx){
    throw std::runtime_error("VJP for RealRMSNorm not implemented yet!");
}

//Dynamic Normalization --------------
void vjp_Dyntanh(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    Node* A = ctx.node->inputs[1].get();
    Node* B = ctx.node->inputs[2].get();
    Node* G = ctx.node->inputs[3].get();
    Tensor h = detail::to_fp32_if_float(*ctx.node->tape[0]);
    Tensor th = OwnTensor::tanh(h);
    Tensor d_th = 1.0f - th * th;
    Tensor G_val = ctx.input(3);
    Tensor A_val = ctx.input(1);
    Tensor X_val = ctx.input(0);

    if (X->requires_grad()) X->grad += ctx.gy * G_val * d_th * A_val;
    if (A->requires_grad()) A->grad += OwnTensor::reduce_sum(ctx.gy * G_val * d_th * X_val, {}, false);
    if (B->requires_grad()) B->grad += OwnTensor::reduce_sum(ctx.gy, {}, false);
    if (G->requires_grad()) G->grad += OwnTensor::reduce_sum(ctx.gy * th, {}, false);
}

//Global Reductions -------------------
void vjp_Sum(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        // ctx.gy is a scalar (1-element tensor).
        // Standard broadcasting rules will expand it to X's shape during addition.
        X->grad += ctx.gy;
    }
}
void vjp_MeanAll(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        float scale = 1.0f / static_cast<float>(X->value.numel());
        // ctx.gy is a scalar. We scale it and let broadcasting handle the expansion.
        X->grad += ctx.gy * scale;
    }
}
//Row-wise Reductions ------------------------
void vjp_RowSum(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        // ctx.gy has reduced dimensions (e.g., [B, 1] or [B]).
        // Implicit broadcasting should handle expansion to [B, N].
        X->grad += ctx.gy;
    }
}

void vjp_RowMax(const VjpContext& ctx){
    throw std::runtime_error("VJP for RowMax not implemented yet!");
}

//Softmax Family ---------------
void vjp_SoftmaxRow(const VjpContext& ctx){
    Node* Z = ctx.node->inputs[0].get();
    if (Z->requires_grad()) {
        Tensor y = detail::to_fp32_if_float(ctx.node->value);
        Tensor dot = OwnTensor::reduce_sum(y * ctx.gy, {-1}, true);
        Z->grad += y * (ctx.gy - dot);
    }
}

void vjp_LogSumExpRow(const VjpContext& ctx){
    Node* Z = ctx.node->inputs[0].get();
    if (Z->requires_grad()) {
        Tensor z_val = ctx.input(0);
        Tensor max_val = OwnTensor::reduce_max(z_val, {-1}, true);
        Tensor exp_z = OwnTensor::exp(z_val - max_val, ag::current_stream());
        Tensor sum_exp_z = OwnTensor::reduce_sum(exp_z, {-1}, true);
        Tensor softmax_z = exp_z / sum_exp_z;
        Z->grad += ctx.gy * softmax_z;
    }
}

//Trigonometric Functions --------------------
void vjp_Sin(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) X->grad += ctx.gy * OwnTensor::cos(ctx.input(0));
}

void vjp_Cos(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) X->grad -= ctx.gy * OwnTensor::sin(ctx.input(0));
}

void vjp_Tan(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor c = OwnTensor::cos(ctx.input(0));
        X->grad += ctx.gy / (c * c);
    }
}

//Hyperbolic Functions ------------------
void vjp_Cosh(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) X->grad += ctx.gy * OwnTensor::sinh(ctx.input(0));
}
void vjp_Sinh(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) X->grad += ctx.gy * OwnTensor::cosh(ctx.input(0));
}

//Inverse Trigonometric Functions --------------
void vjp_Asin(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = ctx.input(0);
        X->grad += ctx.gy / OwnTensor::sqrt(1.0f - x * x, ag::current_stream());
    }
}

void vjp_Acos(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = ctx.input(0);
        X->grad -= ctx.gy / OwnTensor::sqrt(1.0f - x * x, ag::current_stream());
    }
}

void vjp_Atan(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = ctx.input(0);
        X->grad += ctx.gy / (1.0f + x * x);
    }
}

//Inverse Hyperbolic Trigonometric Functions ----------------
void vjp_ASinh(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = ctx.input(0);
        X->grad += ctx.gy / OwnTensor::sqrt(x * x + 1.0f, ag::current_stream());
    }
}

void vjp_ACosh(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = ctx.input(0);
        X->grad += ctx.gy / OwnTensor::sqrt(x * x - 1.0f, ag::current_stream());
    }
}

void vjp_ATanh(const VjpContext& ctx){
    Node* X = ctx.node->inputs[0].get();
    if (X->requires_grad()) {
        Tensor x = ctx.input(0);
        X->grad += ctx.gy / (1.0f - x * x);
    }
}


} // namespace detail

VjpFn vjp_lookup(Op op){
    switch(op){
#define OP(name, arity, str) case Op::name: return &detail::vjp_##name;
#include "ad/detail/ops.def"
#undef OP
        default: return nullptr;
    }
}

} // namespace ag