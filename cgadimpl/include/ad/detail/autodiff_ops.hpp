// =============================================
// cgadimpl/include/ad/detail/autodiff_ops.hpp
// =============================================
// =============================================
// cgadimpl/include/ad/detail/autodiff_ops.hpp
// =============================================
#pragma once
#include <functional>
#include "ad/core/graph.hpp"
#include "ad/core/schema.hpp"

namespace ag {
namespace detail {
    Tensor to_fp32_if_float(const Tensor& t);
}

struct VjpContext {
    Node* node;
    Tensor gy;

    mutable Tensor _cache[4];
    mutable bool _is_cached[4] = {false, false, false, false};

    VjpContext(Node* n, Tensor g) : node(n), gy(g) {}

    Tensor input(int i) const {
        if (i < 4) {
            if (!_is_cached[i]) {
                _cache[i] = detail::to_fp32_if_float(node->inputs[i]->value);
                _is_cached[i] = true;
            }
            return _cache[i];
        }
        return detail::to_fp32_if_float(node->inputs[i]->value);
    }
};

// VJP: given node n and its output upstream grad gy, accumulate grads into parents.

/// @brief look up table which maps the op to its respective backward function call and gives a computed result. 
/// this function gives the accumulated gradient tobkcwrd op to continue the chain derivation 
using VjpFn = void(*)(const VjpContext& ctx);

// JVP: compute tangent for node n given a way to read parent tangents.
// tangent_of(p) must return the tangent T[p] (same shape as p->value).
using JvpFn = Tensor(*)(Node* n, const std::function<const Tensor&(Node*)>& tangent_of);

// Lookup tables (one slot per Op value).
VjpFn vjp_lookup(Op op);
JvpFn jvp_lookup(Op op);
// Optional: expose per-op rule symbols to tests only.


} // namespace ag
#ifdef AG_EXPOSE_AUTODIFF_RULES
namespace ag::detail {
  // Declare all rule functions via the registry
  #define OP(name, arity, str) \
    void   vjp_##name(const VjpContext& ctx); \
    Tensor jvp_##name(Node* n, const std::function<const Tensor&(Node*)>& tangent_of); \
  #include "ad/detail/ops.def"
  #undef OP
} // namespace ag::detail
#endif