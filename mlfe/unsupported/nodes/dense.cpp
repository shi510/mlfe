#include "dense.hpp"
#include "../../math/blas.hpp"
#include "../../utils/assert.hpp"

namespace mlfe { namespace node {

Dense::Dense() : NodeIO<Dense>("Dense") { }

void Dense::InternalInit(Workspace *ws, OperatorContext *oc) {
    runtime_assert(Inputs() == 3,
        std::string("Inputs must be 3(x, w, b).") +
        std::string(" - Your input size : ") +
        std::to_string(Inputs())
    );

    runtime_assert(Outputs() == 1,
        std::string("Outputs must be 1(y).") +
        std::string(" - Your output size : ") +
        std::to_string(Outputs())
    );
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->Get<Tensor>(base->Input(0));
    Tensor *w = ws->Create<Tensor>(base->Input(1));
    Tensor *b = ws->Create<Tensor>(base->Input(2));
    Tensor *y = ws->Create<Tensor>(base->Output(0));
    w->SetTrainable(true);
    b->SetTrainable(true);
    b->SetBias(true);
    
    const int batch = x->Dim(0);
    const int in_neurons = x->Dim(1);
    const int out_neurons = oc->attr->GetParam<int>("Outputs");

    w->Reshape({ out_neurons, in_neurons });
    b->Reshape({ out_neurons });
    y->Reshape({ batch, out_neurons });

    oc->inputs.push_back(x);
    oc->inputs.push_back(w);
    oc->inputs.push_back(b);
    oc->outputs.push_back(y);
}

void Dense::InternalGradientInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->Get<Tensor>(base->Input(0));
    Tensor *w = ws->Get<Tensor>(base->Input(1));
    Tensor *dy = ws->Get<Tensor>(base->Output(0) + "_grad");
    Tensor *dw = ws->Create<Tensor>(base->Input(1) + "_grad");
    Tensor *db = ws->Create<Tensor>(base->Input(2) + "_grad");
    Tensor *dx = ws->Create<Tensor>(base->Input(0) + "_grad");

    const int batch = x->Dim(0);
    const int in_neurons = x->Dim(1);
    const int out_neurons = oc->attr->GetParam<int>("Outputs");

    dw->Reshape({ out_neurons, in_neurons });
    db->Reshape({ out_neurons });
    dx->Reshape({ batch, in_neurons });

    oc->inputs.push_back(x);
    oc->inputs.push_back(w);
    oc->inputs.push_back(dy);
    oc->outputs.push_back(dw);
    oc->outputs.push_back(db);
    oc->outputs.push_back(dx);
}
} // end namespace node
} // end namespace mlfe
