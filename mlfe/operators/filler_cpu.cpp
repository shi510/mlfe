#include "filler.hpp"
#include "../device_context/cpu_context.hpp"
#include "../math/blas.hpp"
#include "../math/functions.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <class DC>
FillOp<DC>::FillOp(
                           OperatorIO &opio,
                           ItemHolder *ih
                           ) : Operator<DC>(opio, ih), rng(math::GetRandomSeed()) {
    runtime_assert(this->inputs.size() == 0,
                   "[Fill Op] inputs.size() == 0.");
    runtime_assert(this->outputs.size() == 1,
                   "[Fill Op] outputs.size() == 1.");
}

template <class DT, class DC>
ConstantFillOp<DT, DC>::ConstantFillOp(
                                                  OperatorIO &opio,
                                                  ItemHolder *ih
                                                  ) : FillOp<DC>(opio, ih) {
    auto y = this->outputs[OutputSchema::y];
    if(opio.param.HasParam("Value")){
        val = opio.param.GetParam<DT>("Value");
    }
    else{
        val = static_cast<DT>(0);
    }
    if(y->IsEmpty()){
        throw std::string("[ConstantFill] Output is empty.");
    }
}

template <class DT, class DC>
void ConstantFillOp<DT, DC>::Compute(){
    auto y = this->outputs[OutputSchema::y];
    DT *ptr = y->template GetPtrMutable<DT>();
    for(int n = 0; n < y->Size(); ++n){
        ptr[n] = val;
    }
}

REGIST_OPERATOR_CPU(ConstantFill_float, ConstantFillOp<float, CPUContext>)
REGIST_OPERATOR_CPU(ConstantFill_double, ConstantFillOp<double, CPUContext>)

template <class DT, class DC>
XavierFillOp<DT, DC>::XavierFillOp(
                                              OperatorIO &opio,
                                              ItemHolder *ih
                                              ) : FillOp<DC>(opio, ih) {
    auto y = this->outputs[OutputSchema::y];
    if(y->IsEmpty()){
        throw std::string("[XaiverFill] Output is empty.");
    }
    scale = std::sqrt(static_cast<DT>(2) / static_cast<DT>(y->Size() / y->Dim(0)));
    uniform = std::uniform_real_distribution<DT>(-scale, scale);
}

template <class DT, class DC>
void XavierFillOp<DT, DC>::Compute(){
    auto y = this->outputs[OutputSchema::y];
    DT *ptr = y->template GetPtrMutable<DT>();
    for(int n = 0; n < y->Size(); ++n){
        ptr[n] = uniform(this->rng);
    }
}

REGIST_OPERATOR_CPU(XavierFill_float, XavierFillOp<float, CPUContext>)
REGIST_OPERATOR_CPU(XavierFill_double, XavierFillOp<double, CPUContext>)

} /* namespace mlfe */
