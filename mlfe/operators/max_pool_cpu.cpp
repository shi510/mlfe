#include <algorithm>
#include "max_pool.hpp"
#include "../device_context/cpu_context.hpp"

namespace mlfe{

template <class DT, class DC>
MaxPoolOp<DT, DC>::MaxPoolOp(
                                        OperatorIO &opio,
                                        ItemHolder *ih
                                        ) : Operator<DC>(opio, ih) {
    runtime_assert(this->inputs.size() == 1,
                   "[MaxPool Op] inputs.size() == 1.");
    runtime_assert(this->outputs.size() == 2,
                   "[MaxPool Op] outputs.size() == 2.");
    const auto x = this->inputs[InputSchema::x];
    auto y = this->outputs[OutputSchema::y];
    auto idx = this->outputs[OutputSchema::idx];
    
    if(opio.param.HasParam("Kernel") &&
       opio.param.HasParam("Stride") &&
       y->IsEmpty() &&
       idx->IsEmpty() &&
       !x->IsEmpty() &&
       x->Dims() == 4){
        kernel = opio.param.GetParam<std::vector<int>>("Kernel");
        stride = opio.param.GetParam<std::vector<int>>("Stride");
        out_h = (x->Dim(2) - kernel[0]) / stride[0] + 1;
        out_w = (x->Dim(3) - kernel[1]) / stride[1] + 1;
        y->template Resize<DT>({x->Dim(0), x->Dim(1), out_h, out_w});
        idx->template Resize<int>(*y);
    }
    else{
        runtime_assert(x->Dims() == 2,
                       "[MaxPool Op] x->Dims() == 2.");
        runtime_assert(y->CompareSizeWith(*idx),
                       "[MaxPool Op] y->CompareSizeWith(idx).");
    }
}

template <class DT, class DC>
void MaxPoolOp<DT, DC>::Compute(){
    const auto x = this->inputs[InputSchema::x];
    auto idx = this->outputs[OutputSchema::idx];
    auto y = this->outputs[OutputSchema::y];
    const DT *x_ptr = x->template GetPtrConst<DT>();
    DT *y_ptr = y->template GetPtrMutable<DT>();
    int *idx_ptr = idx->template GetPtrMutable<int>();
    
    y->template SetByConst<DT>(static_cast<DT>(-FLT_MAX));
    for (int n = 0; n < x->Dim(0); ++n){
        for (int c = 0; c < x->Dim(1); ++c){
            for (int ph = 0; ph < out_h; ++ph){
                for (int pw = 0; pw < out_w; ++pw){
                    int hstart = ph * stride[0];
                    int wstart = pw * stride[1];
                    int hend = std::min<int>(hstart + kernel[0], x->Dim(2));
                    int wend = std::min<int>(wstart + kernel[1], x->Dim(3));
                    const int pool_index = ph * out_w + pw;
                    for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                            const int index = h * x->Dim(3) + w;
                            if (x_ptr[index] > y_ptr[pool_index]) {
                                y_ptr[pool_index] = x_ptr[index];
                                idx_ptr[pool_index] = index;
                            }
                        }
                    }
                }
            }
            x_ptr += x->Dim(2) * x->Dim(3);
            y_ptr += out_h * out_w;
            idx_ptr += out_h * out_w;
        }
    }
}

REGIST_OPERATOR_CPU(MaxPool_float, MaxPoolOp<float, CPUContext>)
REGIST_OPERATOR_CPU(MaxPool_double, MaxPoolOp<double, CPUContext>)

template <class DT, class DC>
MaxPoolGradientOp<DT, DC>::MaxPoolGradientOp(
                                                        OperatorIO &opio,
                                                        ItemHolder *ih
                                                        ) : Operator<DC>(opio, ih) {
    runtime_assert(this->inputs.size() == 3,
                   "[MaxPool Gradient Op] inputs.size() == 3.");
    runtime_assert(this->outputs.size() == 1,
                   "[MaxPool Gradient Op] outputs.size() == 1.");
    
    const auto x = this->inputs[InputSchema::x];
    const auto idx = this->inputs[InputSchema::idx];
    const auto dy = this->inputs[InputSchema::dy];
    auto dx = this->outputs[OutputSchema::dx];
    if(opio.param.HasParam("Kernel") &&
       opio.param.HasParam("Stride") &&
       dx->IsEmpty() &&
       !x->IsEmpty() &&
       !idx->IsEmpty() &&
       !dy->IsEmpty() &&
       x->Dims() == 4 &&
       dy->CompareSizeWith(*idx)
       ){
        kernel = opio.param.GetParam<std::vector<int>>("Kernel");
        stride = opio.param.GetParam<std::vector<int>>("Stride");
        out_h = dy->Dim(2);
        out_w = dy->Dim(3);
        dx->template Resize<DT>(*x);
    }
    else{
        runtime_assert(idx->CompareSizeWith(*dy),
                       "[MaxPool Gradient Op] idx->CompareSizeWith(dy)");
    }
}

template <class DT, class DC>
void MaxPoolGradientOp<DT, DC>::Compute(){
    const auto idx = this->inputs[InputSchema::idx];
    const auto dy = this->inputs[InputSchema::dy];
    auto dx = this->outputs[OutputSchema::dx];
    const DT *dy_ptr = dy->template GetPtrConst<DT>();
    const int *idx_ptr = idx->template GetPtrConst<int>();
    DT *dx_ptr = dx->template GetPtrMutable<DT>();
    
    dx->template SetByConst<DT>(static_cast<DT>(0));
    for (int n = 0; n < dy->Dim(0); ++n) {
        for (int c = 0; c < dy->Dim(1); ++c) {
            for (int ph = 0; ph < out_h; ++ph) {
                for (int pw = 0; pw < out_w; ++pw) {
                    const int index = ph * out_w + pw;
                    const int bottom_index = idx_ptr[index];
                    dx_ptr[bottom_index] += dy_ptr[index];
                }
            }
            dx_ptr += dx->Dim(2) * dx->Dim(3);
            dy_ptr += dy->Dim(2) * dy->Dim(3);
            idx_ptr += idx->Dim(2) * idx->Dim(3);
        }
    }
}

REGIST_OPERATOR_CPU(MaxPool_float_Gradient, MaxPoolGradientOp<float, CPUContext>)
REGIST_OPERATOR_CPU(MaxPool_double_Gradient, MaxPoolGradientOp<double, CPUContext>)

struct MaxPoolGradientIO : public GradientIO{
    OperatorIO GetGradientIO(OperatorIO opio) override{
        OperatorIO opio_grad;
        opio_grad.type = opio.type + "_" + opio.data_type + "_Gradient";
        opio_grad.data_type = opio.data_type;
        opio_grad.inputs.push_back(opio.inputs[0]);
        opio_grad.inputs.push_back(opio.outputs[1]);
        opio_grad.inputs.push_back(opio.outputs[0] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[0] + "_grad");
        opio_grad.param = opio.param;
        
        return opio_grad;
    }
};

REGIST_OPERATOR_GRADIENT_IO(MaxPool, MaxPoolGradientIO);

} /* namespace mlfe */
