#include "cross_entropy.hpp"
#include "../../math/blas.hpp"
#include "../../utils/assert.hpp"

namespace mlfe { namespace node {

SoftmaxCrossEntropyWithLabel::SoftmaxCrossEntropyWithLabel()
    : NodeSchema<SoftmaxCrossEntropyWithLabel>("SoftmaxXentWithLabel") { }

void SoftmaxCrossEntropyWithLabel::InternalInit(Workspace *ws, OperatorContext *oc) {
    runtime_assert(Inputs() == 2,
        std::string("Inputs must be 2(x, label).") +
        std::string(" - Your input size : ") +
        std::to_string(Inputs())
    );
    runtime_assert(Outputs() == 2,
        std::string("Outputs must be 2(prob, loss).") +
        std::string(" - Your output size : ") +
        std::to_string(Outputs())
    );

    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->GetIfNotExistCreate<Tensor>(base->Input(0));
    Tensor *label = ws->GetIfNotExistCreate<Tensor>(base->Input(1));
    Tensor *prob = ws->GetIfNotExistCreate<Tensor>(base->Output(0));
    Tensor *loss = ws->GetIfNotExistCreate<Tensor>(base->Output(1));

    /*
    * batch size.
    */
    const int m = x->Dim(0);

    /*
    * output size.
    */
    const int n = x->Dim(1);

    prob->Reshape({ m, n });
    loss->Reshape({ 1 });

    oc->inputs.push_back(x);
    oc->inputs.push_back(label);
    oc->outputs.push_back(prob);
    oc->outputs.push_back(loss);
}

void SoftmaxCrossEntropyWithLabel::InternalGradientInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->GetIfNotExistCreate<Tensor>(base->Input(0));
    Tensor *label = ws->GetIfNotExistCreate<Tensor>(base->Input(1));
    Tensor *prob = ws->GetIfNotExistCreate<Tensor>(base->Output(0));
    Tensor *loss = ws->GetIfNotExistCreate<Tensor>(base->Output(1));
    Tensor *dx = ws->GetIfNotExistCreate<Tensor>(base->Input(0) + "_grad");

    /*
    * batch size.
    */
    const int m = x->Dim(0);

    /*
    * output size.
    */
    const int n = x->Dim(1);

    dx->Reshape({ m, n });

    oc->inputs.push_back(label);
    oc->inputs.push_back(prob);
    oc->inputs.push_back(loss);
    oc->outputs.push_back(dx);
}

SigmoidCrossEntropy::SigmoidCrossEntropy()
    : NodeSchema<SigmoidCrossEntropy>("SigmoidXent") { }

void SigmoidCrossEntropy::InternalInit(Workspace *ws, OperatorContext *oc) {
    runtime_assert(Inputs() == 2,
        std::string("Inputs must be 2(x, t).") +
        std::string(" - Your input size : ") +
        std::to_string(Inputs())
    );
    runtime_assert(Outputs() == 1,
        std::string("Outputs must be 1(loss).") +
        std::string(" - Your output size : ") +
        std::to_string(Outputs())
    );

    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->GetIfNotExistCreate<Tensor>(base->Input(0));
    Tensor *sigmoid = ws->GetIfNotExistCreate<Tensor>(base->Input(0) + "_sigmoid");
    Tensor *t = ws->GetIfNotExistCreate<Tensor>(base->Input(1));
    Tensor *loss = ws->GetIfNotExistCreate<Tensor>(base->Output(0));

    loss->Reshape({ x->Dim(0) });
    sigmoid->Reshape(x->Shape());

    oc->inputs.push_back(x);
    oc->inputs.push_back(sigmoid);
    oc->inputs.push_back(t);
    oc->outputs.push_back(loss);
}

void SigmoidCrossEntropy::InternalGradientInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->GetIfNotExistCreate<Tensor>(base->Input(0));
    Tensor *sigmoid = ws->GetIfNotExistCreate<Tensor>(base->Input(0) + "_sigmoid");
    Tensor *t = ws->GetIfNotExistCreate<Tensor>(base->Input(1));
    Tensor *loss = ws->GetIfNotExistCreate<Tensor>(base->Output(0));
    Tensor *dx = ws->GetIfNotExistCreate<Tensor>(base->Input(0) + "_grad");

    dx->Reshape(x->Shape());

    oc->inputs.push_back(x);
    oc->inputs.push_back(sigmoid);
    oc->inputs.push_back(t);
    oc->inputs.push_back(loss);
    oc->outputs.push_back(dx);
}

} // end namespace node
} // end namespace mlfe
