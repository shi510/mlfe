#include "accuracy.hpp"

namespace mlfe {namespace node {

Accuracy::Accuracy() : NodeIO<Accuracy>("Accuracy"){ }

void Accuracy::InternalInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *prob = ws->Get<Tensor>(base->Input(0));
    Tensor *label = ws->Get<Tensor>(base->Input(1));
    Tensor *accuracy = ws->Create<Tensor>(base->Output(0));
    accuracy->Reshape({ 1 });
    oc->inputs.push_back(prob);
    oc->inputs.push_back(label);
    oc->outputs.push_back(accuracy);
    std::cout << Name() << " -> " << "InternalInit on Node." << std::endl;
}

void Accuracy::InternalGradientInit(Workspace *ws, OperatorContext *oc) { }

} // end namespace node
} // end namespace mlfe
