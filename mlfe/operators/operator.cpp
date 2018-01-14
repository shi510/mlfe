#include "operator.hpp"

namespace mlfe{

OperatorBase::OperatorBase(
                           OperatorIO &opio,
                           ItemHolder *ih
                           ){
    this->opio = opio;
    this->ih = ih;
}

Item *OperatorBase::Inputs(const int idx) {
    return ih->GetItem(opio.inputs[idx]);
}

Item *OperatorBase::Outputs(const int idx) {
    return ih->GetItem(opio.outputs[idx]);
}

int OperatorBase::Inputs() {
    return opio.inputs.size();
}

int OperatorBase::Outputs() {
    return opio.outputs.size();
}

OperatorIO & OperatorBase::GetOperatorIO(){
    return opio;
}

DEFINE_REGISTRY(
                 OperatorCPU,
                 std::string,
                 std::shared_ptr<OperatorBase>,
                 OperatorIO &,
                 ItemHolder *
                 )

DEFINE_REGISTRY(
                OperatorGradientIO,
                std::string,
                std::shared_ptr<GradientIO>
                )

std::shared_ptr<OperatorBase>
CreateOperator(OperatorIO &opio, ItemHolder *ih){
    auto type = opio.type;
    if (!opio.data_type.empty()) {
        type += "_" + opio.data_type;
    }
    if(!opio.accelerator.empty()){
        type += "_" + opio.accelerator;
    }
    auto op = OperatorCPU()->Create(type, opio, ih);
    return op;
}

std::shared_ptr<OperatorBase>
CreateOperatorGradient(OperatorIO &opio, ItemHolder *ih){
    auto grad = OperatorGradientIO()->Create(opio.type);
    auto opio_grad = grad->GetGradientIO(opio);
    auto op = OperatorCPU()->Create(opio_grad.type, opio_grad, ih);
    return op;
}

} /* namespace mlfe */
