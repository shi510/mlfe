#include "gradient_descent.h"
#include "../core/op_design.h"

namespace mlfe{ namespace optimizer{

GradientDescent::GradientDescent(double learning_rate)
    : lr(learning_rate){}

AppliedOptimizer GradientDescent::Minimize(Tensor loss){
    auto grad_pair = LetGradientFlowBack(loss);
    UpdateRule ur(loss, std::get<0>(grad_pair));
    for(auto pair : std::get<1>(grad_pair)){
        auto dep = OpDependency::Builder("GradientDescent")
            .Input({ "X", pair.first })
            .Input({ "dX", pair.second })
            .Output({ "Y", pair.first })
            .Attr({ "LearningRate", static_cast<float>(lr) })
            .Finish();
        ur.AddRule(dep);
    }
    return ApplyGradientUpdateRule(ur);
}

REGIST_OP(GradientDescent)
    .Input("X", type::float32::string)
    .Input("dX", type::float32::string)
    .Output("Y", type::float32::string)
    .Attr("LearningRate", type::float32::string)
    .Finish();

GradientDescentWithMomentum::GradientDescentWithMomentum(
    double learning_rate,
    double momentum_rate,
    double weight_decay
    ) : lr(learning_rate), mr(momentum_rate), wd(weight_decay){}

AppliedOptimizer GradientDescentWithMomentum::Minimize(Tensor loss){
    auto grad_pair = LetGradientFlowBack(loss);
    UpdateRule ur(loss, std::get<0>(grad_pair));
    for(auto pair : std::get<1>(grad_pair)){
        auto dep = OpDependency::Builder("GradientDescentWithMomentum")
            .Input({ "X", pair.first })
            .Input({ "dX", pair.second })
            .Output({ "Y", pair.first })
            .Attr({ "LearningRate", static_cast<float>(lr) })
            .Attr({ "MomentumRate", static_cast<float>(mr) })
            .Attr({ "WeightDecay", static_cast<float>(wd) })
            .Finish();
        ur.AddRule(dep);
    }
    return ApplyGradientUpdateRule(ur);
}

REGIST_OP(GradientDescentWithMomentum)
    .Input("X", type::float32::string)
    .Input("dX", type::float32::string)
    .Output("Y", type::float32::string)
    .Attr("LearningRate", type::float32::string)
    .Attr("MomentumRate", type::float32::string)
    .Attr("WeightDecay", type::float32::string)
    .Finish();

} // end namespace optimizer
} // end namespace mlfe
