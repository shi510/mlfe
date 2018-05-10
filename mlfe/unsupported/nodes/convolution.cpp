#include "convolution.hpp"

namespace mlfe {namespace node {

Conv::Conv() : NodeIO<Conv>("Conv"){ }


// * Convolution Node
//     - Input Shape = [batch, In Filters, Height, Width]
//     - Weight Shape = [Out Filters, In Filters, Weight_H, Weight_W]
//     - Bias Shape = [Out Filters]
//     - ConvOut Width = (Width + Padding * 2 - Weight_W) / Stride_W + 1
//     - ConvOut Height = (Height + Padding * 2 - Weight_H) / Stride_H + 1
//     - Output Shape = [batch, Out Filters, ConvOut Height, ConvOut Width]
void Conv::InternalInit(Workspace *ws, OperatorContext *oc) {
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->Get<Tensor>(base->Input(0));
    Tensor *w = ws->Create<Tensor>(base->Input(1));
    Tensor *b = ws->Create<Tensor>(base->Input(2));
    Tensor *y = ws->Create<Tensor>(base->Output(0));
    auto filter = oc->attr->GetParam<int>("Filters");
    auto kernel = oc->attr->GetParam<std::vector<int>>("Kernel");
    auto stride = oc->attr->GetParam<std::vector<int>>("Stride");
    auto padding = oc->attr->GetParam<std::vector<int>>("Padding");
    int batch = x->Dim(0);
    int out_h = (x->Dim(3) + padding[1] * 2 - kernel[1]) / stride[1] + 1;
    int out_w = (x->Dim(2) + padding[0] * 2 - kernel[0]) / stride[0] + 1;
    w->SetTrainable(true);
    b->SetTrainable(true);
    b->SetBias(true);

    w->Reshape({ filter, x->Dim(1), kernel[0], kernel[1] });
    b->Reshape({ filter });
    y->Reshape({ batch, filter, out_h, out_w });
    
    oc->inputs.push_back(x);
    oc->inputs.push_back(w);
    oc->inputs.push_back(b);
    oc->outputs.push_back(y);
}

// * Convolution Gradient Node
//     - Input Shape = [batch, In Filters, Height, Width]
//     - Weight Shape = [Out Filters, In Filters, Weight_H, Weight_W]
//     - Bias Shape = [Out Filters]
//     - Output Gradient Shape = Output Shape
//     - Weight Gradient Shape = Weight Shape
//     - Biase Gradient Shape = Bias Shape
//     - Input Gradient Shape = Input Shape
void Conv::InternalGradientInit(Workspace *ws, OperatorContext *oc) { 
    Node *base = reinterpret_cast<Node *>(this);
    Tensor *x = ws->Get<Tensor>(base->Input(0));
    Tensor *w = ws->Get<Tensor>(base->Input(1));
    Tensor *dy = ws->Get<Tensor>(base->Output(0) + "_grad");
    Tensor *dw = ws->Create<Tensor>(base->Input(1) + "_grad");
    Tensor *db = ws->Create<Tensor>(base->Input(2) + "_grad");
    Tensor *dx = ws->Create<Tensor>(base->Input(0) + "_grad");
    auto filter = oc->attr->GetParam<int>("Filters");
    auto kernel = oc->attr->GetParam<std::vector<int>>("Kernel");
    auto stride = oc->attr->GetParam<std::vector<int>>("Stride");
    auto padding = oc->attr->GetParam<std::vector<int>>("Padding");
    int batch = x->Dim(0);
    int out_h = (x->Dim(3) + padding[1] * 2 - kernel[1]) / stride[1] + 1;
    int out_w = (x->Dim(2) + padding[0] * 2 - kernel[0]) / stride[0] + 1;

    dw->Reshape({ filter, x->Dim(1), kernel[0], kernel[1] });
    db->Reshape({ filter });
    dx->Reshape({ x->Dim(0), x->Dim(1), x->Dim(2), x->Dim(3) });

    oc->inputs.push_back(x);
    oc->inputs.push_back(w);
    oc->inputs.push_back(dy);
    oc->outputs.push_back(dw);
    oc->outputs.push_back(db);
    oc->outputs.push_back(dx);
}

} // end namespace node
} // end namespace mlfe
