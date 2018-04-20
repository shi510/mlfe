#include "initializer.hpp"

namespace mlfe {namespace node {

ConstantInit::ConstantInit() 
    : NodeIO<ConstantInit>("ConstantInit"){ }

void ConstantInit::InternalInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->Get<Tensor>(base->Input(0));
    oc->inputs.push_back(x);
    std::cout << Name() << " -> " << "InternalInit on Node." << std::endl;
}

void ConstantInit::InternalGradientInit(Workspace *ws, OperatorContext *oc) {
    std::cout << Name() << " -> " << "InternalInit on NodeGradient." << std::endl;
}

XavierInit::XavierInit() 
    : NodeIO<XavierInit>("XavierInit") { }

void XavierInit::InternalInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->Get<Tensor>(base->Input(0));

    oc->inputs.push_back(x);
    std::cout << Name() << " -> " << "InternalInit on Node." << std::endl;
}

void XavierInit::InternalGradientInit(Workspace *ws, OperatorContext *oc) {
    std::cout << Name() << " -> " << "InternalInit on NodeGradient." << std::endl;
}

} // end namespace node
} // end namespace mlfe
