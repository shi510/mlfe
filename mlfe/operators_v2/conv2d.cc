#include "mlfe/operators_v2/conv2d.h"
#include "mlfe/operators_v2/utils.h"
#include <iostream>

namespace mlfe{
namespace operators_v2{

Tensor conv2d(
    Tensor x,                     // N, H, W, C
    Tensor kernel,                // Out, Height, Width, In
    std::vector<int32_t> strides, // Height, Width
    std::vector<int32_t> paddings // Height, Width
    )
{
    if(x.dims() != 4){
        std::cout<<"conv2d: input dimension should be 4. ";
        std::cout<<"but "<<x.dims()<<std::endl;
        std::runtime_error("Wrong parameter of conv2d");
    }
    if(kernel.dims() != 4){
        std::cout<<"conv2d: kernel dimension should be 4. ";
        std::cout<<"but "<<kernel.dims()<<std::endl;
        std::runtime_error("Wrong parameter of conv2d");
    }
    int out_h = utils::calc_conv_output(
        x.shape()[1], kernel.shape()[0], strides[0], paddings[0]);
    int out_w = utils::calc_conv_output(
        x.shape()[2], kernel.shape()[1], strides[1], paddings[1]);
    auto y =
        functional::create_variable({x.shape()[0], out_h, out_w, kernel.shape()[3]});
    auto gm_x = [=](Tensor dy){
        conv2d_input_bwd_kernel::fn(kernel, dy, x.grad_v2(), strides, paddings);
    };
    auto gm_k = [=](Tensor dy){
        conv2d_kernel_bwd_kernel::fn(x, dy, kernel.grad_v2(), strides, paddings);
    };
    call<conv2d_fwd_kernel>(
        marker::I(x, kernel),
        marker::O(y)(gm_x, gm_k),
        strides, paddings);
    return y;
}

Tensor conv2d(
    Tensor x,
    Tensor kernel,
    std::vector<int> strides,
    bool same_out
    )
{
    int ph = utils::calc_conv_same_output_padding_size(
        x.shape()[1], kernel.shape()[0], strides[0]);
    int pw = utils::calc_conv_same_output_padding_size(
        x.shape()[2], kernel.shape()[1], strides[1]);
    return conv2d(x, kernel, strides, {ph, pw});
}

} // namespace operators_v2
} // namespace mlfe
