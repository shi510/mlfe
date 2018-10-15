#include "optimizer.h"
#include "../core/graph.h"
#include "../operators/initializer.h"
#include "../operators/basic_arithmetics.h"
#include <algorithm>

namespace mlfe{ namespace optimizer{
using Opt = Optimizer;
using OptUR = Optimizer::UpdateRule;
using AO = AppliedOptimizer;

Optimizer::TensorPairs Optimizer::compute_gradient(const Tensor root){
    using TensorUmap = std::unordered_map<Tensor, std::vector<Tensor>>;
    TensorUmap dy_collector;
    TensorPairs gpair;
    // top-down seuqnce
    auto v_list = visit_bfs(root);
    // sort by execution order.
    std::sort(v_list.begin(), v_list.end(), [](Tensor v1, Tensor v2){
        return v1.get_exec_order() > v2.get_exec_order();
    });
    // root gradient is 1.
    dy_collector[root].push_back(functional::Constant(1, root.Shape()));
    bwd_op_deps.clear();
    bwd_op_deps.push_back(dy_collector[root][0].get_dep());
    for(auto &var : v_list){
        auto op_name = var.get_dep().Name();
        if(op_name != "Unknown"){
            auto helper = GradientHelperRegistry::Get();
            auto op_grad = helper->GetHelper(op_name, var.get_dep().Context());
            //add all partial gradients and propagate down.
            auto dy = functional::add_n(dy_collector[var]);
            auto input_grad = op_grad->compute_gradient(var, dy);
            for(auto &it : input_grad){
                dy_collector[it.first].push_back(it.second);
                gpair.push_back({it.first, it.second});
            }
            if(dy_collector[var].size() > 1){
                bwd_op_deps.push_back(dy.get_dep());
            }
            bwd_op_deps.push_back(op_grad->get_opdep());
        }
    }
    return gpair;
}

Optimizer::OpDeps Optimizer::get_op_deps() const{
    return bwd_op_deps;
}

AO Opt::ApplyGradientUpdateRule(OptUR ur){
    AO ao(ur);
    return ao;
}

Opt::UpdateRule::UpdateRule(Tensor target, OpDeps bwd_op_deps)
    : opt_target(target), _bwd_op_deps(bwd_op_deps){}

Opt::UpdateRule::UpdateRule(Tensor target, Tensor dx)
    : opt_target(target){}

void Opt::UpdateRule::AddRule(OpDependency od){
    _update_op_deps.push_back(od);
}

Tensor AO::Target() const{
    return rule.opt_target;
}

std::vector<OpDependency> AO::get_bwd_op_deps() const{
    return rule._bwd_op_deps;
}

std::vector<OpDependency> AO::get_update_op_deps() const{
    return rule._update_op_deps;
}

AO::AppliedOptimizer(Optimizer::UpdateRule ur)
    : rule(ur){}

} // end namespace optimizer
} // end namespace mlfe
