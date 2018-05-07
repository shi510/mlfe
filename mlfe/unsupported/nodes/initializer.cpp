#include "initializer.hpp"

namespace mlfe {namespace node {

ConstantInit::ConstantInit() 
    : NodeIO<ConstantInit>("ConstantInit"){ }

void ConstantInit::InternalInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->Get<Tensor>(base->Input(0));
    oc->inputs.push_back(x);
}

void ConstantInit::InternalGradientInit(Workspace *ws, OperatorContext *oc) {}

XavierInit::XavierInit() 
    : NodeIO<XavierInit>("XavierInit") { }

void XavierInit::InternalInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->Get<Tensor>(base->Input(0));

    oc->inputs.push_back(x);
}

void XavierInit::InternalGradientInit(Workspace *ws, OperatorContext *oc) {}

} // end namespace node
} // end namespace mlfe
