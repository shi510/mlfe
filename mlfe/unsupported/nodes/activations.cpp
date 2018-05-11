#include "activations.hpp"

namespace mlfe {namespace node {

Activation::Activation() : NodeSchema<Activation>("Activation"){ }

void Activation::InternalInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    auto x = ws->Get<Tensor>(base->Input(0));
    auto y = ws->Create<Tensor>(base->Output(0));
    y->Reshape(x->Shape());
    oc->inputs.push_back(x);
    oc->outputs.push_back(y);
}

void Activation::InternalGradientInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    auto x = ws->Get<Tensor>(base->Input(0));
    auto y = ws->Get<Tensor>(base->Output(0));
    auto dy = ws->Get<Tensor>(base->Output(0) + "_grad");
    auto dx = ws->Create<Tensor>(base->Input(0) + "_grad");
    dx->Reshape(x->Shape());
    oc->inputs.push_back(x);
    oc->inputs.push_back(y);
    oc->inputs.push_back(dy);
    oc->outputs.push_back(dx);
}

} // end namespace node
} // end namespace mlfe
