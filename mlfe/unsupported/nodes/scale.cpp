#include "scale.hpp"

namespace mlfe {namespace node {

Scale::Scale()
    : NodeSchema<Scale>("Scale"){ }

void Scale::InternalInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->Get<Tensor>(base->Input(0));
    Tensor *y = ws->GetIfNotExistCreate<Tensor>(base->Output(0));
    if (y->Size() == 0) {
        y->Reshape(x->Shape());
    }
    oc->inputs.push_back(x);
    oc->outputs.push_back(y);
}

void Scale::InternalGradientInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *dy = ws->Get<Tensor>(base->Output(0) + "_grad");
    Tensor *dx = ws->GetIfNotExistCreate<Tensor>(base->Input(0) + "_grad");
    if (dx->Size() == 0) {
        dx->Reshape(dy->Shape());
    }
    oc->inputs.push_back(dy);
    oc->outputs.push_back(dx);
}

} // end namespace node
} // end namespace mlfe
