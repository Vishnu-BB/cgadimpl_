// =====================
// file: include/ag/autodiff.hpp (declarations only)
// =====================
#pragma once
#include <unordered_map>
#include "ad/ops/ops.hpp"


namespace ag {
namespace detail{
    Tensor to_fp32_if_float(const Tensor& t);
}

void zero_grad(const Value& root);
void backward (const Value& root, const Tensor* grad_seed=nullptr, bool enable_parallel=false);

Tensor jvp (const Value& root, const std::unordered_map<Node*, Tensor>& seed);


} // namespace ag