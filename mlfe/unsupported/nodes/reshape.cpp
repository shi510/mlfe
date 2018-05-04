#include "reshape.hpp"

namespace mlfe {namespace node {

Reshape::Reshape() : NodeIO<Reshape>("Reshape"){ }

void Reshape::InternalInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    auto dim = oc->attr->GetParam<std::vector<int>>("Dim");
    auto x = ws->Get<Tensor>(base->Input(0));
    auto y = ws->Create<Tensor>(base->Output(0));
    *y = *x;
    y->Reshape(dim);
}

void Reshape::InternalGradientInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->Get<Tensor>(base->Input(0));
    Tensor *dy = ws->Get<Tensor>(base->Output(0) + "_grad");
    Tensor *dx = ws->Create<Tensor>(base->Input(0) + "_grad");
    *dx = *dy;
    dx->Reshape(x->Shape());
}

} // end namespace node
} // end namespace mlfe
