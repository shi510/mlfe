#include "adadelta.h"
#include "../core/op_design.h"

namespace mlfe{ namespace optimizer{


AdaDelta::AdaDelta(
    double learning_rate,
    double momentum_rate,
    double epsilon
    ) : lr(learning_rate), mr(momentum_rate), eps(epsilon){}

AppliedOptimizer AdaDelta::Minimize(Tensor loss){
    auto grad_pair = LetGradientFlowBack(loss);
    UpdateRule ur(loss, std::get<0>(grad_pair));
    for(auto pair : std::get<1>(grad_pair)){
        auto dep = OpDependency::Builder("AdaDelta")
            .Input(std::make_tuple("X", pair.first))
            .Input(std::make_tuple("dX", pair.second))
            .Output(std::make_tuple("Y", pair.first))
            .Attr({ "LearningRate", static_cast<float>(lr) })
            .Attr({ "MomentumRate", static_cast<float>(mr) })
            .Attr({ "Epsilon", static_cast<float>(eps) })
            .Finish();
        ur.AddRule(dep);
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
