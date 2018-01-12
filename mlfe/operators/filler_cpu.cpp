#include "filler.hpp"
#include "../device_context/cpu_context.hpp"
#include "../math/blas.hpp"
#include "../math/functions.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <>
FillOp<CPUContext>::FillOp(
                           OperatorIO &opio,
                           ItemHolder *ih
                           ) : Operator<CPUContext>(opio, ih), rng(math::GetRandomSeed()) {
    runtime_assert(inputs.size() == 0,
                   "[Fill Op] inputs.size() == 0.");
    runtime_assert(outputs.size() == 1,
                   "[Fill Op] outputs.size() == 1.");
}

template <>
ConstantFillOp<float, CPUContext>::ConstantFillOp(
                                                  OperatorIO &opio,
                                                  ItemHolder *ih
                                                  ) : FillOp<CPUContext>(opio, ih) {
    auto y = outputs[OutputSchema::y];
    if(opio.param.HasParam("Value")){
        val = opio.param.GetParam<float>("Value");
    }
    else{
        val = static_cast<float>(0);
    }
    if(y->IsEmpty()){
        throw std::string("[ConstantFill] Output is empty.");
    }
}

template <>
void ConstantFillOp<float, CPUContext>::Compute(){
    auto y = outputs[OutputSchema::y];
    float *ptr = y->GetPtrMutable<float>();
    for(int n = 0; n < y->Size(); ++n){
        ptr[n] = val;
    }
}

REGIST_OPERATOR_CPU(ConstantFill, ConstantFillOp<float, CPUContext>)

template <>
XavierFillOp<float, CPUContext>::XavierFillOp(
                                              OperatorIO &opio,
                                              ItemHolder *ih
                                              ) : FillOp<CPUContext>(opio, ih) {
    auto y = outputs[OutputSchema::y];
    if(y->IsEmpty()){
        throw std::string("[XaiverFill] Output is empty.");
    }
    scale = std::sqrt(static_cast<float>(2) / static_cast<float>(y->Size() / y->Dim(0)));
    uniform = std::uniform_real_distribution<float>(-scale, scale);
}

template <>
void XavierFillOp<float, CPUContext>::Compute(){
    auto y = outputs[OutputSchema::y];
    float *ptr = y->GetPtrMutable<float>();
    for(int n = 0; n < y->Size(); ++n){
        ptr[n] = uniform(this->rng);
    }
}

REGIST_OPERATOR_CPU(XavierFill, XavierFillOp<float, CPUContext>)

} /* namespace mlfe */
