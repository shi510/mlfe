#include "optimizer.h"
#include <algorithm>

namespace mlfe{ namespace optimizer{
using Opt = Optimizer;
using OptUR = Optimizer::UpdateRule;
using AO = AppliedOptimizer;

GradientHelper::HelperOut Opt::LetGradientFlowBack(Tensor loss){
    auto ghr = GradientHelperRegistry::Get();
    auto op_deps = loss.OpDependencies();
    Tensor dy = loss;
    std::reverse(op_deps.begin(), op_deps.end());
    GradientHelper::GradientPairs grad_pairs;
    for(auto dep : op_deps){
        auto helper = ghr->GetHelper(dep.Name(), dep.Context());
        auto grads = helper->Get(dy);
        dy = std::get<0>(grads);
        for(auto &pair : std::get<1>(grads)){
            grad_pairs.push_back(pair);
        }
    }
    return std::make_tuple(dy, grad_pairs);
}

AO Opt::ApplyGradientUpdateRule(OptUR ur){
    AO ao(ur);
    return ao;
}

Opt::UpdateRule::UpdateRule(Tensor target, Tensor dx)
    : opt_target(target), opt_dx(dx){}

void Opt::UpdateRule::AddRule(OpDependency od){
    deps.push_back(od);
}

Tensor AO::Target() const{
    return rule.opt_target;
}

Tensor AO::InputGrad() const{
    return rule.opt_dx;
}

std::vector<OpDependency> AO::OpDependencies() const{
    return rule.deps;
}

AO::AppliedOptimizer(Optimizer::UpdateRule ur)
    : rule(ur){}

} // end namespace optimizer
} // end namespace mlfe
