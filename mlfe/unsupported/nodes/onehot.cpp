#include "onehot.hpp"

namespace mlfe {namespace node {

OneHot::OneHot() : NodeIO<OneHot>("OneHot"){ }

void OneHot::InternalInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *label = ws->Get<Tensor>(base->Input(0));
    Tensor *onehot = ws->Create<Tensor>(base->Output(0));
    int dim = oc->attr->GetParam<int>("Dim");
    onehot->Reshape({ label->Dim(0), dim });
    oc->inputs.push_back(label);
    oc->outputs.push_back(onehot);
}

void OneHot::InternalGradientInit(Workspace *ws, OperatorContext *oc) { }

} // end namespace node
} // end namespace mlfe
