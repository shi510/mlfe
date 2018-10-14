#include "gradient_descent.h"
#include "../core/op_design.h"

namespace mlfe{ namespace optimizer{

GradientDescent::GradientDescent(double learning_rate)
    : lr(learning_rate){}

AppliedOptimizer GradientDescent::Minimize(Tensor loss){
    auto grad_pair = compute_gradient(loss);
    UpdateRule ur(loss, bwd_op_deps);
    for(auto pair : grad_pair){
        if(pair.first.get_trainable() == true){
            auto dep = OpDependency::Builder("GradientDescent")
                .Input(pair.first)
                .Input(pair.second)
                .Output(pair.first)
                .Attr({"LearningRate", static_cast<float>(lr)})
                .Finish();
            ur.AddRule(dep);
        }
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
    auto grad_pair = compute_gradient(loss);
    UpdateRule ur(loss, bwd_op_deps);
    for(auto pair : grad_pair){
        if(pair.first.get_trainable() == true){
            auto dep = OpDependency::Builder("GradientDescentWithMomentum")
                .Input(pair.first)
                .Input(pair.second)
                .Output(pair.first)
                .Attr({"LearningRate", static_cast<float>(lr)})
                .Attr({"MomentumRate", static_cast<float>(mr)})
                .Attr({"WeightDecay", static_cast<float>(wd)})
                .Finish();
            ur.AddRule(dep);
        }
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
