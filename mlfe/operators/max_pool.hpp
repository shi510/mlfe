#ifndef __MAX_POOL_OP_HPP__
#define __MAX_POOL_OP_HPP__

#include <cfloat>
#include "operator.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <class DataType, class DeviceContext>
class MaxPoolOp final : public Operator<DeviceContext>{
public:
    explicit MaxPoolOp(
                       std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs,
                       std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs,
                       ParamDef param = ParamDef()
                       ) : Operator<DeviceContext>(inputs, outputs, param) {
        runtime_assert(inputs.size() == 1, "Input size must be 1(x).");
        runtime_assert(outputs.size() == 2, "Output size must be 2(y, idx).");
        const auto x = this->Input(InputSchema::x);
        auto y = this->Output(OutputSchema::y);
        auto idx = this->Output(OutputSchema::idx);
        
        if(this->GetParam().GetParamByName("Kernel", kernel) &&
           this->GetParam().GetParamByName("Stride", stride) &&
           y->IsEmpty() &&
           idx->IsEmpty() &&
           !x->IsEmpty() &&
           x->Dims() == 4){
            out_h = (x->Dim(2) - kernel[0]) / stride[0] + 1;
            out_w = (x->Dim(3) - kernel[1]) / stride[1] + 1;
            y->template Reshape<DataType>({x->Dim(0), x->Dim(1), out_h, out_w});
            idx->template ReshapeLike<int>(y);
        }
        else{
            runtime_assert(x->Dims() == 2, "x's dim size must be 2.");
            runtime_assert(y->CompareSizeWith(idx) , "y's dim(1) must be same with w's dim(0).");
            runtime_assert(y->Dim(1), "y's dim(1) must be same with w's dim(0).");
        }
    }
    
    void Compute() override {
        const auto x = this->Input(InputSchema::x);
        auto idx = this->Output(OutputSchema::idx);
        auto y = this->Output(OutputSchema::y);
        const DataType *x_ptr = x->template GetPtrConst<DataType>();
        DataType *y_ptr = y->template GetPtrMutable<DataType>();
        int *idx_ptr = idx->template GetPtrMutable<int>();
        
        y->template SetByConst<DataType>(static_cast<DataType>(-FLT_MAX));
        for (int n = 0; n < x->Dim(0); ++n){
            for (int c = 0; c < x->Dim(1); ++c){
                for (int ph = 0; ph < out_h; ++ph){
                    for (int pw = 0; pw < out_w; ++pw){
                        int hstart = ph * stride[0];
                        int wstart = pw * stride[1];
                        int hend = std::min(hstart + kernel[0], x->Dim(2));
                        int wend = std::min(wstart + kernel[1], x->Dim(3));
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
    
private:
    enum InputSchema{x};
    enum OutputSchema{y, idx};
    std::vector<int> kernel;
    std::vector<int> stride;
    int out_h, out_w;
};

template <class DataType, class DeviceContext>
class MaxPoolGradientOp final : public Operator<DeviceContext>{
public:
    explicit MaxPoolGradientOp(
                               std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs,
                               std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs,
                               ParamDef param = ParamDef()
                               ) : Operator<DeviceContext>(inputs, outputs, param) {
        runtime_assert(inputs.size() == 3, "Input size must be 3(x, idx, dy).");
        runtime_assert(outputs.size() == 1, "Output size must be 3(dx).");
        
        const auto x = this->Input(InputSchema::x);
        const auto idx = this->Input(InputSchema::idx);
        const auto dy = this->Input(InputSchema::dy);
        auto dx = this->Output(OutputSchema::dx);
        if(this->GetParam().GetParamByName("Kernel", kernel) &&
           this->GetParam().GetParamByName("Stride", stride) &&
           dx->IsEmpty() &&
           !x->IsEmpty() &&
           !idx->IsEmpty() &&
           !dy->IsEmpty() &&
           x->Dims() == 4 &&
           dy->CompareSizeWith(idx)
           ){
            out_h = dy->Dim(2);
            out_w = dy->Dim(3);
            dx->template ReshapeLike<DataType>(x);
        }
        else{
            runtime_assert(idx->CompareSizeWith(dy), "[MaxPoolGradient : idx's size must be same with dy.");
        }
    }
    
    void Compute() override {
        const auto x = this->Input(InputSchema::x);
        const auto idx = this->Input(InputSchema::idx);
        const auto dy = this->Input(InputSchema::dy);
        auto dx = this->Output(OutputSchema::dx);
        const DataType *dy_ptr = dy->template GetPtrConst<DataType>();
        const int *idx_ptr = idx->template GetPtrConst<int>();
        DataType *dx_ptr = dx->template GetPtrMutable<DataType>();
        
        dx->template SetByConst<DataType>(static_cast<DataType>(0));
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
    
private:
    enum InputSchema{x, idx, dy};
    enum OutputSchema{dx};
    std::vector<int> kernel;
    std::vector<int> stride;
    int out_h, out_w;
};

} /* namespace mlfe */
#endif /* __MAX_POOL_OP_HPP__ */
