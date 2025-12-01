// =====================
// file: cgadimpl/src/ops.cpp
// =====================
#include "ad/ops.hpp"
#include "ad/nodeops.hpp" // Include the new node-level declarations
#include "ad/runtime.hpp"

namespace ag {

    Value add(const Value& a, const Value& b){ 
        return Value(detail::add_nodeops(a.node, b.node)); 
    }

    Value sub(const Value& a, const Value& b){ 
        
        return Value(detail::sub_nodeops(a.node, b.node)); 
    }

    Value mul(const Value& a, const Value& b){ 
        return Value(detail::mul_nodeops(a.node, b.node)); 
    }

    Value div(const Value& a, const Value& b){ 
        
        return Value(detail::div_nodeops(a.node, b.node)); 
    }

    Value sign(const Value& a, const Value& b){ 
        return Value(detail::sign_nodeops(a.node)); 
    }

    Value sinh(const Value& a, const Value& b){ 
        return Value(detail::sinh_nodeops(a.node)); 
    }

    Value cosh(const Value& a, const Value& b){ 
        return Value(detail::cosh_nodeops(a.node)); 
    }

    Value sin(const Value& a, const Value& b){ 
        return Value(detail::sin_nodeops(a.node)); 
    }

    Value cos(const Value& a, const Value& b){ 
        return Value(detail::cos_nodeops(a.node)); 
    }


    Value flomul(const Value& a, float b){ 
        return Value(detail::flomul_nodeops(a.node, b));
    }

    Value floadd(const Value& a, float b){ 
        return Value(detail::floadd_nodeops(b, a.node));
    }

    Value flodiv(const Value& a, float b){ 
        return Value(detail::flodiv_nodeops(b, a.node));
    }

    Value relu(const Value& x){     
        return Value(detail::relu_nodeops(x.node));
    }

    Value reci(const Value& x){    
        return Value(detail::reci_nodeops(x.node));
    }

    Value cos(const Value& x){ 
        return Value(detail::cos_nodeops(x.node));
    }

    Value sinh(const Value& x){ 
        return Value(detail::sinh_nodeops(x.node));
    }

    Value cosh(const Value& x){ 
        return Value(detail::cosh_nodeops(x.node));
    }

    Value sigatt(const Value& a, const Value& b, const Value& c, const Value& d){ 
        return Value(detail::sigatt_nodeops(a.node, b.node, c.node, d.node));
    }

    Value linear(const Value& a, const Value& b, const Value& c){ 
        return Value(detail::linear_nodeops(a.node, b.node, c.node)); 
    }

    Value moewe(const Value& x, const Value& w, const Value& b){ 
        return Value(detail::moewe_nodeops(x.node, w.node, b.node));
    }

    Value matmul(const Value& a, const Value& b){ 
         return Value(detail::matmul_nodeops(a.node, b.node)); 
    }

    Value fmab(const Value& a, const Value& b, const Value& c){ 
        return Value(detail::fmab_nodeops(a.node, b.node, c.node)); 
    }


    Value attention(const Value& a, const Value& b, const Value& c, const Value& d){ 
    return Value(detail::attention_nodeops(a.node, b.node, c.node, d.node));
    }


    Value alibiatt(const Value& a, const Value& b, const Value& c, const Value& d, float m) { 
    return Value(detail::alibiatt_nodeops(a.node, b.node, c.node, d.node, m));
}

    Value swiglu(const Value& x, const Value& a, const Value& b, const Value& c, const Value& d){ 
    return Value(detail::swiglu_nodeops(x.node, a.node, b.node, c.node, d.node));
    }

    Value sum(const Value& x){ 
        return Value(detail::sum_nodeops(x.node));
    }

    Value transpose(const Value& x){ 
        return Value(detail::transpose_nodeops(x.node));
    }

    Value exp(const Value& x){ 
        return Value(detail::exp_nodeops(x.node));
    }
    
    Value log(const Value& x){ 
        return Value(detail::log_nodeops(x.node));
    }

    Value mish(const Value& x){ 
        return Value(detail::mish_nodeops(x.node));
    }
    
    Value tanh(const Value& x){ 
        return Value(detail::tanh_nodeops(x.node));
    }
    
    Value reluatt(const Value& a, const Value& b, const Value& c, const Value& d){ 
    return Value(detail::reluatt_nodeops(a.node, b.node, c.node, d.node));
    }

    Value sigmoid(const Value& x){ 
        return Value(detail::sigmoid_nodeops(x.node));
    }
    
    Value softplus(const Value& x){ 
        return Value(detail::softplus_nodeops(x.node));
    }

    Value gaus(const Value& x){ 
        return Value(detail::gaus_nodeops(x.node));
    }
    
    Value gelu(const Value& x){ 
        return Value(detail::gelu_nodeops(x.node));
    }

    Value gcu(const Value& x){ 
        return Value(detail::gcu_nodeops(x.node));
    }
    
    Value silu(const Value& x){ 
        return Value(detail::silu_nodeops(x.node));
    }

    Value parcon(const Value& x){ 
        return Value(detail::parcon_nodeops(x.node));
    }

    Value lisht(const Value& x){ 
        return Value(detail::lisht_nodeops(x.node));
    }
    
    Value leaky_relu(const Value& x, float alpha){ 
        return Value(detail::leaky_relu_nodeops(x.node, alpha));
    }

    Value rowsum(const Value& x){ 
        return Value(detail::rowsum_nodeops(x.node));
    }

    Value rowmax(const Value& x){ 
        return Value(detail::rowmax_nodeops(x.node));
    }

    Value rms(const Value& x){ 
        return Value(detail::rms_nodeops(x.node));
    }

    Value realrms(const Value& x, float g){ 
        return Value(detail::realrms_nodeops(x.node, g));
    }

    Value laynor(const Value& x){ 
        return Value(detail::laynor_nodeops(x.node));
    }

    Value relaynor(const Value& x, float b, float g){ 
        return Value(detail::relaynor_nodeops(x.node, b, g));
    }
    
    Value mean_all(const Value& x){ 
        return Value(detail::mean_all_nodeops(x.node));
    }

    Value dyntanh(const Value& x, float a, float b, float g){ 
        return Value(detail::dyntanh_nodeops(x.node, a, b, g));
    }
    
    Value softmax_row(const Value& z){ 
        return Value(detail::softmax_row_nodeops(z.node));
    }
    
    Value logsumexp_row(const Value& z){ 
        return Value(detail::logsumexp_row_nodeops(z.node));
    }

    Value mambassm(const Value& z, const Value& a, const Value& b, const Value& c, const Value& d){
        return Value(detail::mambassm_nodeops(z.node, a.node, b.node, c.node, d.node)); 
    }

    Value cross_entropy_with_logits(const Value& logits, const Value& onehot){
        return Value(detail::cross_entropy_with_logits_nodeops(logits.node, onehot.node));
    }
    
    Value kldivergence(const Value& logits, const Value& onehot){
        return Value(detail::kldivergence_nodeops(logits.node, onehot.node));
    }
    
    Value mse_loss(const Value& pred, const Value& target) {
        return Value(detail::mse_loss_nodeops(pred.node, target.node));
    }

    Value mae_loss(const Value& pred, const Value& target) {
        return Value(detail::mae_loss_nodeops(pred.node, target.node));
    }
}// namespace ag