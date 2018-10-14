#include "adadelta.h"
#include "../core/op_design.h"

namespace mlfe{ namespace optimizer{


AdaDelta::AdaDelta(
    double learning_rate,
    double momentum_rate,
    double epsilon
    ) : lr(learning_rate), mr(momentum_rate), eps(epsilon){}

AppliedOptimizer AdaDelta::Minimize(Tensor loss){
    auto grad_pair = compute_gradient(loss);
    UpdateRule ur(loss, bwd_op_deps);
    for(auto pair : grad_pair){
        if(pair.first.get_trainable() == true){
            auto dep = OpDependency::Builder("AdaDelta")
                .Input(pair.first)
                .Input(pair.second)
                .Output(pair.first)
                .Attr({"LearningRate", static_cast<float>(lr)})
                .Attr({"MomentumRate", static_cast<float>(mr)})
                .Attr({"Epsilon", static_cast<float>(eps)})
                .Finish();
            ur.AddRule(dep);
        }
    }
    return ApplyGradientUpdateRule(ur);
}

REGIST_OP(AdaDelta)
    .Input("X", type::float32::string)
    .Input("dX", type::float32::string)
    .Output("Y", type::float32::string)
    .Attr("LearningRate", type::float32::string)
    .Attr("MomentumRate", type::float32::string)
    .Attr("Epsilon", type::float32::string)
    .Finish();

} // end namespace optimizer
} // end namespace mlfe
