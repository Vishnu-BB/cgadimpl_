#include "ad/runtime/jit_compiler.hpp"
#include "ad/ops/nodeops.hpp" 
#include "TensorLib.h"
#include "ad/core/mlir_emitter.hpp"
#include "Compiler/API/NovaCompilerAPI.h"
#include "mlir/IR/BuiltinOps.h"
#include <unordered_map>
#include <variant>
#include <iostream>
#include <cassert>
#include <sstream>

namespace ag::jit {

using ag::Op;
using ag::Node;
using ag::Value;

// ===================================================================
// JIT Compiler Implementation
// ===================================================================

struct Compiled::Impl {
    Plan plan;

    // --- helpers for replay ---
    static const Tensor& as_ref(const Arg& a,
                                const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& params,
                                const std::vector<Tensor>& slots,
                                Tensor& tmp) {
        if (std::holds_alternative<ArgInput>(a))  return *inputs[std::get<ArgInput>(a).idx];
        if (std::holds_alternative<ArgParam>(a))  return *params[std::get<ArgParam>(a).idx];
        if (std::holds_alternative<ArgSlot>(a))   return slots[std::get<ArgSlot>(a).slot];
        // literal: copy into tmp to return a ref
        const Tensor& lit = std::get<ArgLit>(a).t;
        tmp = lit;
        return tmp;
    }

    static Tensor apply(Op op, const std::vector<const Tensor*>& a) {
        // a.size() equals op_arity(op), except literals we materialized as tensors
        switch(op){
            case Op::Add:        return *a[0] + *a[1];
            case Op::Sub:        return *a[0] - *a[1];
            case Op::Mul:        return *a[0] * *a[1];
            case Op::Div:        return *a[0] / *a[1];
            case Op::Abs:        return OwnTensor::abs(*a[0], (cudaStream_t)ag::current_stream());

            // Unary operators now use the free functions from the OwnTensor namespace.
            case Op::Transpose:  return a[0]->transpose(-2, -1);
            case Op::Relu:     { cudaStream_t stream = (cudaStream_t)ag::current_stream(); return (*a[0] + OwnTensor::abs(*a[0], stream)) * 0.5f;}
            case Op::Exp:        return OwnTensor::exp(*a[0]);
            case Op::Log:        return OwnTensor::log(*a[0]);
            case Op::Tanh:       return OwnTensor::tanh(*a[0]);
            
            case Op::MatMul:     return OwnTensor::matmul(*a[0], *a[1]);

            // Reductions need to be updated to the new API
            case Op::Sum: return OwnTensor::reduce_sum(*a[0]);
            case Op::RowSum: return OwnTensor::reduce_sum(*a[0], {1}, true);
            case Op::RowMax: return OwnTensor::reduce_max(*a[0], {1}, true);
            case Op::MeanAll: return OwnTensor::reduce_mean(*a[0]);

            case Op::GELU: {
                const float c1 = 0.7978845608f; // sqrt(2.0f / M_PI)
                const float c2 = 0.044715f;
                Tensor x3 = (*a[0]) * (*a[0]) * (*a[0]);
                Tensor u = ((*a[0]) + x3 * c2) * c1;
                return (*a[0]) * (1.0f + OwnTensor::tanh(u)) * 0.5f;
            }
            case Op::Sigmoid:    return 1.0f / (1.0f + OwnTensor::exp((*a[0]) * -1.0f));
            case Op::SiLU: {
                Tensor s = 1.0f / (1.0f + OwnTensor::exp((*a[0]) * -1.0f));
                return (*a[0]) * s;
            }
            case Op::LeakyRelu: {
                // a[1] is alpha
                float alpha = a[1]->to_cpu().data<float>()[0];
                cudaStream_t stream = (cudaStream_t)ag::current_stream();
                // LeakyRelu(x) = pos_part + alpha * neg_part
                // pos_part = (x + abs(x)) * 0.5
                // neg_part = (x - abs(x)) * 0.5
                Tensor x = *a[0];
                Tensor abs_x = OwnTensor::abs(x, stream);
                Tensor pos_part = (x + abs_x) * 0.5f;
                Tensor neg_part = (x - abs_x) * 0.5f;
                return pos_part + (neg_part * alpha);
            }
            case Op::Softplus:   return OwnTensor::log(1.0f + OwnTensor::exp(*a[0]));
            case Op::CeWithLogits: {
                // a[0] is logits, a[1] is target
                const Tensor& Z = *a[0];
                const Tensor& Y = *a[1];
                Tensor max_val = OwnTensor::reduce_max(Z, {-1}, true);
                Tensor z_shifted = Z - max_val;
                Tensor log_sum_exp = OwnTensor::log(OwnTensor::reduce_sum(OwnTensor::exp(z_shifted), {-1}, true));
                Tensor log_sm = z_shifted - log_sum_exp;
                Tensor prod = Y * log_sm;
                Tensor sum_prod = OwnTensor::reduce_sum(prod, {-1});
                return OwnTensor::reduce_mean(sum_prod * -1.0f);
            }

            case Op::Leaf: default: {
                // Shouldn't get called for Leaf
                assert(false && "apply(): unexpected op");
                return *a[0];
            }
        }
    }

    bool run(const std::vector<Tensor*>& inputs,
             const std::vector<Tensor*>& params,
             std::vector<Tensor>& outputs) const {
        if (!plan.sig.matches(inputs, params)) return false;

        std::vector<Tensor> slots(plan.num_slots);
        
        for (const Step& st : plan.steps) {
            if (st.out_slot >= 0) {
                slots[st.out_slot] = Tensor(OwnTensor::Shape{st.out_meta.shape}, st.out_meta.dtype, st.out_meta.device, false);
            }
        }

        // Execute
        for (const Step& st : plan.steps) {
            std::vector<const Tensor*> args; args.reserve(st.args.size());
            
            std::vector<Tensor> tmp_keep; tmp_keep.reserve(st.args.size());
            for (const Arg& a : st.args) {
                if (std::holds_alternative<ArgLit>(a)) {
                    tmp_keep.emplace_back(std::get<ArgLit>(a).t);
                    args.push_back(&tmp_keep.back());
                } else {
                    Tensor dummy;
                    args.push_back(&as_ref(a, inputs, params, slots, dummy));
                }
            }
            Tensor y = apply(st.op, args);
            slots[st.out_slot] = std::move(y);
        }

        outputs.clear();
        for (int slot : plan.out_slots) {
            outputs.push_back(slots[slot]);
        }
        return true;
    }
};

static bool is_in(const std::unordered_map<Node*,int>& m, Node* n){ return m.find(n)!=m.end(); }

// --- Helpers for string-based MLIR emission (Fallback) ---
static std::string dtypeToMLIR(Dtype dt) {
    switch (dt) {
        case OwnTensor::Dtype::Float32:  return "f32";
        case OwnTensor::Dtype::Float16:  return "f16";
        case OwnTensor::Dtype::Bfloat16: return "bf16";
        case OwnTensor::Dtype::Int32:    return "i32";
        case OwnTensor::Dtype::Int64:    return "i64";
        default:                        return "unknown";
    }
}

static std::string shapeToMLIR(const std::vector<int64_t>& shape) {
    std::string s;
    for (int64_t dim : shape) {
        s += std::to_string(dim) + "x";
    }
    return s;
}

static std::string opToNovaOp(Op op) {
    switch (op) {
        case Op::Add:       return "nova.add";
        case Op::Mul:       return "nova.mul";
        case Op::MatMul:    return "nova.matmul"; 
        case Op::Sum:       return "nova.reduce<sum>";
        case Op::MeanAll:   return "nova.reduce<mean>";
        default:            return "nova.unknown_op";
    }
}

static std::string emitMLIR(const Plan& plan) {
    std::stringstream ss;
    ss << "func.func @main(";
    size_t arg_idx_counter = 0;

    auto print_arg_meta = [&](const std::vector<TensorMetadata>& metas) {
        for (size_t i = 0; i < metas.size(); ++i) {
            const auto& meta = metas[i];
            ss << "%arg" << arg_idx_counter++ << ": tensor<" 
               << shapeToMLIR(meta.shape) << dtypeToMLIR(meta.dtype) << ">";
            if (i < metas.size() - 1 || !plan.sig.param_meta.empty()) ss << ", ";
        }
    };

    print_arg_meta(plan.sig.in_meta);
    if (!plan.sig.param_meta.empty()) {
        ss << ", ";
        print_arg_meta(plan.sig.param_meta);
    }
    ss << ") -> (";
    
    for (size_t i = 0; i < plan.out_slots.size(); ++i) {
        int slot = plan.out_slots[i];
        // Find meta for this slot
        const TensorMetadata* meta = nullptr;
        for (const auto& st : plan.steps) {
            if (st.out_slot == slot) {
                meta = &st.out_meta;
                break;
            }
        }
        
        if (meta) {
             ss << "tensor<" << shapeToMLIR(meta->shape) << dtypeToMLIR(meta->dtype) << ">";
        }
        if (i < plan.out_slots.size() - 1) ss << ", ";
    }
    ss << ") {\n";

    std::unordered_map<int, std::string> slot_to_var_name;
    std::unordered_map<int, TensorMetadata> slot_to_meta;

    for (const auto& st : plan.steps) {
        slot_to_meta[st.out_slot] = st.out_meta;
    }

    for (size_t i = 0; i < plan.steps.size(); ++i) {
        const auto& st = plan.steps[i];
        std::string result_var = "%v" + std::to_string(i);
        slot_to_var_name[st.out_slot] = result_var;
        ss << "  " << result_var << " = " << opToNovaOp(st.op) << " ";
        std::vector<std::string> arg_names;
        std::vector<std::string> arg_types;

        for (const auto& arg : st.args) {
            std::visit([&](auto&& a) {
                using T = std::decay_t<decltype(a)>;
                if constexpr (std::is_same_v<T, ArgInput> || std::is_same_v<T, ArgParam>) {
                    int arg_idx = a.idx; 
                    const auto& meta = (std::is_same_v<T, ArgInput>) ? plan.sig.in_meta[arg_idx] : plan.sig.param_meta[arg_idx];
                    int base_idx = (std::is_same_v<T, ArgInput>) ? 0 : plan.sig.in_meta.size();
                    arg_names.push_back("%arg" + std::to_string(base_idx + arg_idx));
                    arg_types.push_back("tensor<" + shapeToMLIR(meta.shape) + dtypeToMLIR(meta.dtype) + ">");
                } else if constexpr (std::is_same_v<T, ArgSlot>) {
                    arg_names.push_back(slot_to_var_name.at(a.slot));
                    const auto& meta = slot_to_meta.at(a.slot);
                    arg_types.push_back("tensor<" + shapeToMLIR(meta.shape) + dtypeToMLIR(meta.dtype) + ">");
                } else if constexpr (std::is_same_v<T, ArgLit>) {
                    arg_names.push_back("const_lit"); 
                    arg_types.push_back("tensor<f32>");
                }
            }, arg);
        }

        for (size_t j = 0; j < arg_names.size(); ++j) {
            ss << arg_names[j];
            if (j < arg_names.size() - 1) ss << ", ";
        }
        ss << ": ";
        for (size_t j = 0; j < arg_types.size(); ++j) {
            ss << arg_types[j];
            if (j < arg_types.size() - 1) ss << ", ";
        }
        ss << " -> tensor<" << shapeToMLIR(st.out_meta.shape) << dtypeToMLIR(st.out_meta.dtype) << ">\n";
    }

    ss << "  return ";
    for (size_t i = 0; i < plan.out_slots.size(); ++i) {
        ss << slot_to_var_name.at(plan.out_slots[i]);
        if (i < plan.out_slots.size() - 1) ss << ", ";
    }
    ss << " : ";
    for (size_t i = 0; i < plan.out_slots.size(); ++i) {
        int slot = plan.out_slots[i];
        const auto& return_meta = slot_to_meta.at(slot);
        ss << "tensor<" << shapeToMLIR(return_meta.shape) << dtypeToMLIR(return_meta.dtype) << ">";
        if (i < plan.out_slots.size() - 1) ss << ", ";
    }
    ss << "\n}\n";
    return ss.str();
}

static std::vector<Value> get_symbolic_grads(const Value& loss, const std::vector<Value>& params) {
    auto order = topo_from(loss.node.get());
    std::unordered_map<Node*, Value> grads;

    // Seed: dL/dL = 1.0 (Same shape as loss, usually scalar)
    Tensor one_t = OwnTensor::Tensor::ones(loss.shape(), ag::options(loss.val()));
    grads[loss.node.get()] = make_tensor(one_t, "loss_grad_seed");

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;
        if (grads.find(n) == grads.end()) continue;
        Value gy = grads[n];

        auto add_grad = [&](Node* node, Value g) {
            if (!node || !node->requires_grad()) return;
            if (grads.count(node)) grads[node] = grads[node] + g;
            else grads[node] = g;
        };

        switch (n->op) {
            case Op::Add:
                add_grad(n->inputs[0].get(), gy);
                add_grad(n->inputs[1].get(), gy);
                break;
            case Op::Sub:
                add_grad(n->inputs[0].get(), gy);
                add_grad(n->inputs[1].get(), gy * -1.0f);
                break;
            case Op::Mul:
                add_grad(n->inputs[0].get(), gy * Value(n->inputs[1]));
                add_grad(n->inputs[1].get(), gy * Value(n->inputs[0]));
                break;
            case Op::Div: {
                Value a(n->inputs[0]), b(n->inputs[1]);
                add_grad(a.node.get(), gy / b);
                add_grad(b.node.get(), gy * (a * -1.0f) / (b * b));
                break;
            }
            case Op::MatMul:
                add_grad(n->inputs[0].get(), matmul(gy, transpose(Value(n->inputs[1]))));
                add_grad(n->inputs[1].get(), matmul(transpose(Value(n->inputs[0])), gy));
                break;
            case Op::Sum:
                add_grad(n->inputs[0].get(), gy); // Sum gradient just broadcasts back
                break;
            case Op::MeanAll: {
                float scale = 1.0f / (float)n->inputs[0]->value.numel();
                add_grad(n->inputs[0].get(), gy * scale);
                break;
            }
            case Op::Exp:
                add_grad(n->inputs[0].get(), gy * Value(n->shared_from_this()));
                break;
            case Op::Log:
                add_grad(n->inputs[0].get(), gy / Value(n->inputs[0]));
                break;
            case Op::Tanh: {
                Value y(n->shared_from_this());
                add_grad(n->inputs[0].get(), gy * (1.0f - y * y));
                break;
            }
            case Op::Sigmoid: {
                Value y(n->shared_from_this());
                add_grad(n->inputs[0].get(), gy * y * (1.0f - y));
                break;
            }
            case Op::Relu: {
                // Approximate dy/dx = (x > 0 ? 1 : 0)
                Value x(n->inputs[0]);
                Value mask = (sign(x, x) + 1.0f) * 0.5f; 
                add_grad(n->inputs[0].get(), gy * mask);
                break;
            }
            default:
                break;
        }
    }

    std::vector<Value> param_grads;
    for (const auto& p : params) {
        if (grads.count(p.node.get())) {
            param_grads.push_back(grads[p.node.get()]);
        } else {
            // No path from loss to this param - return zeros
            param_grads.push_back(make_tensor(OwnTensor::Tensor::zeros(p.shape(), ag::options(p.val())), "zero_grad"));
        }
    }
    return param_grads;
}

Compiled compile(const std::vector<Value>& outputs,
                 const std::vector<Value>& inputs,
                 const std::vector<Value>& params,
                 const CompileOptions&) {
    std::unordered_map<Node*,int> in_ix, par_ix;
    for (size_t i = 0; i < inputs.size(); ++i) in_ix[inputs[i].node.get()] = i;
    for (size_t i = 0; i < params.size(); ++i) par_ix[params[i].node.get()] = i;

    Plan plan;
    plan.sig.in_meta.reserve(inputs.size());
    for (const auto& v: inputs) {
        plan.sig.in_meta.push_back({v.shape(), v.val().dtype(), v.val().device()});
    }
    plan.sig.param_meta.reserve(params.size());
    for (const auto& v: params) {
        plan.sig.param_meta.push_back({v.shape(), v.val().dtype(), v.val().device()});
    }

    // Collect all nodes needed for all outputs
    std::vector<Node*> order;
    std::unordered_set<Node*> seen;
    for (const auto& out : outputs) {
        auto sub_order = topo_from(out.node.get());
        for (Node* n : sub_order) {
            if (seen.find(n) == seen.end()) {
                order.push_back(n);
                seen.insert(n);
            }
        }
    }
    
    // Simple topological sort might not be enough if outputs depend on each other or have shared nodes
    // but topo_from already produces an order. We should merge them correctly.
    // Let's use a simpler approach: collect all reachable nodes and then do one topo sort.
    seen.clear();
    order.clear();
    std::function<void(Node*)> collect = [&](Node* n) {
        if (!n || seen.count(n)) return;
        seen.insert(n);
        for (auto& in : n->inputs) collect(in.get());
        order.push_back(n);
    };
    for (const auto& out : outputs) collect(out.node.get());

    std::unordered_map<Node*,int> slot_of;
    slot_of.reserve(order.size());

    for (Node* n : order) {
        if (n->op == Op::Leaf) continue;
        Step st;
        st.op = n->op;
        st.out_meta = {n->shape(), n->value.dtype(), n->value.device()};
        st.out_slot = plan.num_slots++;
        slot_of[n] = st.out_slot;

        st.args.reserve(n->inputs.size());
        for (auto& pin : n->inputs) {
            Node* p = pin.get();
            if (p->op == Op::Leaf) {
                if (is_in(in_ix, p))        st.args.push_back(ArgInput{ in_ix[p] });
                else if (is_in(par_ix, p))  st.args.push_back(ArgParam{ par_ix[p] });
                else                        st.args.push_back(ArgLit{ p->value });
            } else {
                st.args.push_back(ArgSlot{ slot_of.at(p) });
            }
        }
        plan.steps.push_back(std::move(st));
    }
    
    for (const auto& out : outputs) {
        plan.out_slots.push_back(slot_of.at(out.node.get()));
    }

    std::string generated_mlir_opbuilder;
    mlir::OwningOpRef<mlir::ModuleOp> in_memory_module;
    std::shared_ptr<mlir::MLIRContext> context;
    
    try {
        MLIREmitter emitter;
        context = emitter.getContext();
        auto [module, mlirStr] = emitter.emitModule(plan);
        generated_mlir_opbuilder = mlirStr;
        in_memory_module = std::move(module);
        std::cout << "\n=== MLIR Generated via OpBuilder ===\n" << generated_mlir_opbuilder << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: MLIR OpBuilder emission failed: " << e.what() << "\n";
    }

    std::string generated_mlir_string = emitMLIR(plan);
    if (generated_mlir_opbuilder.empty()) {
        std::cout << "\n=== MLIR Generated via String (Fallback) ===\n" << generated_mlir_string << std::endl;
    }

    Compiled c;
    c.p = std::make_shared<Compiled::Impl>();
    c.p->plan = std::move(plan);
    c.mlir_source = std::move(generated_mlir_string);
    
    if (in_memory_module) {
        try {
            mlir::nova::NovaCompilerAPI compiler;
            mlir::nova::CompilerOptions options;
            options.runFullPipeline = true;
            auto compileResult = compiler.compileString(generated_mlir_opbuilder, "", options);
            if (compileResult.success) {
                generated_mlir_opbuilder = compileResult.output;
                std::cout << "\n=== Optimized MLIR Generated via NovaCompilerAPI ===\n" << generated_mlir_opbuilder << std::endl;
            } else {
                std::cerr << "Warning: NovaCompilerAPI pipeline failed: " << compileResult.errorMessage << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: NovaCompilerAPI integration failed: " << e.what() << "\n";
        }

        auto* module_ptr = new mlir::OwningOpRef<mlir::ModuleOp>(std::move(in_memory_module));
        c.mlir_module = std::shared_ptr<void>(module_ptr, [context](void* p) {
            delete static_cast<mlir::OwningOpRef<mlir::ModuleOp>*>(p);
        });
    }

    c.mlir_module_str = std::move(generated_mlir_opbuilder);
    return c;
}

bool Compiled::run(const std::vector<Tensor*>& inputs,
                   const std::vector<Tensor*>& params,
                   std::vector<Tensor>& outputs) const {
    return p->run(inputs, params, outputs);
}

const std::string& Compiled::getMLIRSource() const {
    return mlir_source;
}

void* Compiled::getMLIRModule() const {
    if (mlir_module) {
        auto* module_ptr = static_cast<mlir::OwningOpRef<mlir::ModuleOp>*>(mlir_module.get());
        if (module_ptr && module_ptr->get()) {
            return module_ptr->get();
        }
    }
    return nullptr;
}

Compiled compile_with_backward(const Value& loss,
                               const std::vector<Value>& inputs,
                               const std::vector<Value>& params,
                               const CompileOptions& opts) {
    std::vector<Value> grads = get_symbolic_grads(loss, params);
    
    std::vector<Value> all_roots;
    all_roots.push_back(loss);
    for (const auto& g : grads) all_roots.push_back(g);
    
    return compile(all_roots, inputs, params, opts);
}

} // namespace ag::jit
