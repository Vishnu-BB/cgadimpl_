#include "nn/nn.hpp"
#include <cmath>

namespace ag::nn {

void Module::to(Device dev) {
    for (auto& p : params_) {
        if (p.node && p.node->value.device() != dev) {
            p.node->value = p.node->value.to(dev);
            if (p.node->grad.is_valid()) {
                p.node->grad = p.node->grad.to(dev);
            }
        }
    }
}

void Module::zero_grad() {
    for (auto& p : params_) {
        if (p.node && p.node->grad.is_valid()) {
            p.node->grad = Tensor::zeros(p.node->grad.shape(), 
                TensorOptions().with_dtype(p.node->grad.dtype()).with_device(p.node->grad.device()));
        }
    }
}

Linear::Linear(int in_features, int out_features, Device dev) {
    float k = 1.0f / std::sqrt((float)in_features);
    
    Tensor Wt = Tensor::randn(Shape{{in_features, out_features}}, 
        TensorOptions().with_device(dev).with_req_grad(true)) * k;
    Tensor bt = Tensor::zeros(Shape{{1, out_features}}, 
        TensorOptions().with_device(dev).with_req_grad(true));
    
    W = make_tensor(Wt, "linear_W");
    b = make_tensor(bt, "linear_b");
    
    params_.push_back(W);
    params_.push_back(b);
}

Value Linear::operator()(Value input) {
    return matmul(input, W) + b;
}

Sequential::Sequential(const std::vector<Module*>& modules) : layers_(modules) {
    for (auto* m : layers_) {
        for (auto& p : m->parameters()) {
            params_.push_back(p);
        }
    }
}

Value Sequential::operator()(Value x) {
    for (auto* layer : layers_) {
        x = (*layer)(x);
    }
    return x;
}

Value ReLU::operator()(Value input) {
    return relu(input);
}

} // namespace ag::nn