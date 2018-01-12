#ifndef __CONVOLUTION_BASE_OP_HPP__
#define __CONVOLUTION_BASE_OP_HPP__
#include "operator.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <class DeviceContext>
class ConvolutionBaseOp : public Operator<DeviceContext>{
public:
    void Compute() override = 0;
    
protected:
    explicit ConvolutionBaseOp(
                               OperatorIO &opio,
                               ItemHolder *ih
                               ) : Operator<DeviceContext>(opio, ih){
            runtime_assert(opio.param.HasParam("Stride"),
                           "[Convolution With Eigen Op] Not Found : Stride Param.");
            runtime_assert(opio.param.HasParam("Padding"),
                           "[Convolution With Eigen Op] Not Found : Padding Param.");
            stride = opio.param.GetParam<std::vector<int>>("Stride");
            padding = opio.param.GetParam<int>("Padding");
        }
    
    int OutHeightSize(){
        int height = this->inputs[0]->Dim(2);
        return (height + 2 * padding - kernel_size[0]) / stride[0] + 1;
    }
    
    int OutWidthSize(){
        int width = this->inputs[0]->Dim(3);
        return (width + 2 * padding - kernel_size[1]) / stride[1] + 1;
    }
    
    std::vector<int> kernel_size;
    std::vector<int> stride;
    int filters;
    int padding;
};

} /* namespace mlfe */
#endif /* __CONVOLUTION_BASE_OP_HPP__ */
