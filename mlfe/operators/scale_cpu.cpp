#include "scale.hpp"
#include "../device_context/cpu_context.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <>
ScaleOp<float, CPUContext>::ScaleOp(
                                    OperatorIO &opio,
                                    ItemHolder *ih
                                    ) : Operator<CPUContext>(opio, ih) {
    runtime_assert(inputs.size() == 1,
                   "[Scale Op] inputs.size() == 1.");
    runtime_assert(outputs.size() == 1,
                   "[Scale Op] outputs.size() == 1.");
    const auto x = inputs[InputSchema::x];
    auto y = outputs[OutputSchema::y];
    
    runtime_assert(opio.param.HasParam("Scale"),
                   "[Scale Op] Not found Scale param.");
    if(y->IsEmpty() &&
       !x->IsEmpty()
       ){
        scaler = opio.param.GetParam<float>("Scale");
        y->Resize<float>(*x);
    }
    else{
        runtime_assert(x->Dims() == y->Dims(),
                       "[Scale Op] x->Dims() == y->Dims().");
    }
}

template <>
void ScaleOp<float, CPUContext>::Compute(){
    const auto x = inputs[InputSchema::x];
    auto y = outputs[OutputSchema::y];
    math::scal<float, CPUContext>(
                                  x->Size(),
                                  scaler,
                                  x->GetPtrConst<float>(),
                                  y->GetPtrMutable<float>()
                                  );
}

REGIST_OPERATOR_CPU(Scale, ScaleOp<float, CPUContext>)

struct ScaleGradientIO : public GradientIO{
    OperatorIO GetGradientIO(OperatorIO opio) override{
        OperatorIO opio_grad;
        opio_grad.type = opio.type;
        opio_grad.inputs.push_back(opio.outputs[0] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[0] + "_grad");
        opio_grad.param.Add("Scale", 1.f);
        
        return opio_grad;
    }
};

REGIST_OPERATOR_GRADIENT_IO(Scale, ScaleGradientIO);

} /* namespace mlfe */
