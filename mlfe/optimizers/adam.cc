#include "adam.h"
#include "../core/op_design.h"

namespace mlfe{ namespace optimizer{


Adam::Adam(
    double learning_rate,
    double epsilon,
    double beta1,
    double beta2
    ) : lr(learning_rate), b1(beta1), b2(beta2), eps(epsilon){}

AppliedOptimizer Adam::Minimize(Tensor loss){
    auto grad_pair = LetGradientFlowBack(loss);
    UpdateRule ur(loss, std::get<0>(grad_pair));
    for(auto pair : std::get<1>(grad_pair)){
        auto dep = OpDependency::Builder("Adam")
            .Input(std::make_tuple("X", pair.first))
            .Input(std::make_tuple("dX", pair.second))
            .Output(std::make_tuple("Y", pair.first))
            .Attr({ "LearningRate", static_cast<float>(lr) })
            .Attr({ "Beta1", static_cast<float>(b1) })
            .Attr({ "Beta2", static_cast<float>(b2) })
            .Attr({ "Epsilon", static_cast<float>(eps) })
            .Finish();
        ur.AddRule(dep);
    }
    return ApplyGradientUpdateRule(ur);
}

REGIST_OP(Adam)
    .Input("X", type::float32::string)
    .Input("dX", type::float32::string)
    .Output("Y", type::float32::string)
    .Attr("LearningRate", type::float32::string)
    .Attr("Beta1", type::float32::string)
    .Attr("Beta2", type::float32::string)
    .Finish();

} // end namespace optimizer
} // end namespace mlfe
