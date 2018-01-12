#include "relu.hpp"
#include "../device_context/cpu_context.hpp"

namespace mlfe{

template <>
ReluOp<float, CPUContext>::ReluOp(
                                  OperatorIO &opio,
                                  ItemHolder *ih
                                  ) : Operator<CPUContext>(opio, ih) {
    runtime_assert(inputs.size() == 1,
                   "[Relu Op] inputs.size() == 1.");
    runtime_assert(outputs.size() == 1,
                   "[Relu Op] outputs.size() == 1.");
    const auto x = inputs[InputSchema::x];
    auto y = outputs[OutputSchema::y];
    inplace = false;
    if(opio.param.HasParam("Inplace")){
        inplace = opio.param.GetParam<bool>("Inplace");
    }
    
    if(y->IsEmpty() &&
       !x->IsEmpty()
       ){
        if(inplace){
            *y = *x;
        }
        else{
            y->Resize<float>(*x);
        }
    }
    else{
        runtime_assert(x->Dim(0) == y->Dim(0),
                       "[Relu Op] x->Dim(0) == y->Dim(0).");
    }
}

template <>
void ReluOp<float, CPUContext>::Compute(){
    const auto x = inputs[InputSchema::x];
    auto y = outputs[OutputSchema::y];
    math::ReluFunction<float, CPUContext>(
                                          x->Size(),
                                          x->GetPtrConst<float>(),
                                          y->GetPtrMutable<float>()
                                          );
}

REGIST_OPERATOR_CPU(Relu, ReluOp<float, CPUContext>)


template <>
ReluGradientOp<float, CPUContext>::ReluGradientOp(
                                                  OperatorIO &opio,
                                                  ItemHolder *ih
                                                  ) : Operator<CPUContext>(opio, ih) {
    runtime_assert(inputs.size() == 2,
                   "[Relu Gradient Op] inputs.size() == 2.");
    runtime_assert(outputs.size() == 1,
                   "[Relu Gradient Op] outputs.size() == 1.");
    
    const auto x = inputs[InputSchema::x];
    const auto dy = inputs[InputSchema::dy];
    auto dx = outputs[OutputSchema::dx];
    inplace = false;
    if(opio.param.HasParam("Inplace")){
        inplace = opio.param.GetParam<bool>("Inplace");
    }
    
    if(dx->IsEmpty() &&
       !x->IsEmpty()
       ){
        if(inplace){
            *dx = *dy;
        }
        else{
            dx->Resize<float>(*x);
        }
    }
    else{
        runtime_assert(dx->CompareSizeWith(*x),
                       "[Relu Gradient Op] dx->CompareSizeWith(x).");
    }
    
}

template <>
void ReluGradientOp<float, CPUContext>::Compute(){
    const auto x = inputs[InputSchema::x];
    const auto dy = inputs[InputSchema::dy];
    auto dx = outputs[OutputSchema::dx];
    math::ReluGradientFunction<float, CPUContext>(
                                                  dy->Size(),
                                                  x->GetPtrConst<float>(),
                                                  dy->GetPtrConst<float>(),
                                                  dx->GetPtrMutable<float>()
                                                  );
}

REGIST_OPERATOR_CPU(Relu_Gradient, ReluGradientOp<float, CPUContext>)

struct ReluGradientIO : public GradientIO{
    OperatorIO GetGradientIO(OperatorIO opio) override{
        OperatorIO opio_grad;
        opio_grad.type = opio.type + "_Gradient";
        opio_grad.inputs.push_back(opio.inputs[0]);
        opio_grad.inputs.push_back(opio.outputs[0] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[0] + "_grad");
        opio_grad.param = opio.param;
        
        return opio_grad;
    }
};

REGIST_OPERATOR_GRADIENT_IO(Relu, ReluGradientIO);

} /* namespace mlfe */
