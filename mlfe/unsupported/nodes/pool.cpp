#include "pool.hpp"

namespace mlfe {namespace node {

MaxPool::MaxPool() : NodeIO<MaxPool>("MaxPool"){ }

void MaxPool::InternalInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->Get<Tensor>(base->Input(0));
    Tensor *y = ws->Create<Tensor>(base->Output(0));
    Tensor *idx = ws->Create<Tensor>(base->Output(0) + "_idx");
    auto kernel = oc->attr->GetParam<std::vector<int>>("Kernel");
    auto stride = oc->attr->GetParam<std::vector<int>>("Stride");
    int batch = x->Dim(0);
    int filter = x->Dim(1);
    int out_h = (x->Dim(2) - kernel[0]) / stride[0] + 1;
    int out_w = (x->Dim(3) - kernel[1]) / stride[1] + 1;

    idx->Reshape({ batch, filter, out_h, out_w });
    y->Reshape({ batch, filter, out_h, out_w });

    oc->inputs.push_back(x);
    oc->outputs.push_back(y);
    oc->outputs.push_back(idx);
}

void MaxPool::InternalGradientInit(Workspace *ws, OperatorContext *oc) { 
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->Get<Tensor>(base->Input(0));
    Tensor *y = ws->Get<Tensor>(base->Output(0));
    Tensor *idx = ws->Get<Tensor>(base->Output(0) + "_idx");
    Tensor *dy = ws->Get<Tensor>(base->Output(0) + "_grad");
    Tensor *dx = ws->Create<Tensor>(base->Input(0) + "_grad");

    dx->Reshape(x->Shape());
    
    oc->inputs.push_back(x);
    oc->inputs.push_back(y);
    oc->inputs.push_back(idx);
    oc->inputs.push_back(dy);
    oc->outputs.push_back(dx);
}

} // end namespace node
} // end namespace mlfe
