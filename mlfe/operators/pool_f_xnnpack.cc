#include "mlfe/core/op_algo.h"
#include "mlfe/core/device.h"
#include "xnnpack.h"
#include <iostream>

namespace mlfe{
namespace algorithm_xnnpack{

/*
 * Data order: NHWC
 * Input: X
 * Output: Y
 */
template <class Tp>
class MaxPool : public OpAlgo{
using T = typename Tp::T;
public:
    MaxPool(OpAlgoContext *oac) : OpAlgo(oac, "MaxPool"){
        using IntVec = std::vector<type::int32::T>;
        if(xnn_status_success != xnn_initialize(nullptr)){
            std::cout<<"Fail xnn_initialize"<<std::endl;
            exit(1);
        }
        strides = oac->get_attr<IntVec>("stride");
        pads = oac->get_attr<IntVec>("padding");
        kernel = oac->get_attr<IntVec>("kernel");
        y = oac->get_output(0);
        x = oac->get_input(0);
        resize();
    }

    void resize() override{
        int batch = x.shape()[0];
        int in_h = x.shape()[1];
        int in_w = x.shape()[2];
        int in_c = x.shape()[3];
        auto status = xnn_create_max_pooling2d_nhwc_f32(
            /*top=*/pads[0], /*right=*/pads[1],
            /*bottom=*/pads[0], /*left=*/pads[1],
            /*kernel_h=*/kernel[0], /*kernel_h=*/kernel[1],
            /*stride_h=*/strides[0], /*stride_w=*/strides[1],
            /*dilation height=*/1, /*dilation width=*/1,
            /*input_channels=*/in_c,
            /*input_pixel_stride=*/in_c,
            /*output_pixel_stride=*/in_c,
            /*output_min=*/std::numeric_limits<T>::min(),
            /*output_max=*/std::numeric_limits<T>::max(),
            /*same_out_padding=*/0,
            &maxpool_op);
        if(xnn_status_success != status){
            std::cout<<"Fail xnn_create_convolution2d_nhwc_f32"<<std::endl;
            exit(1);
        }
        status = xnn_setup_max_pooling2d_nhwc_f32(
            maxpool_op,
            batch, in_h, in_w,
            x.device_data<T>(),
            y.mutable_device_data<T>(),
            nullptr);
        if(xnn_status_success != status){
            std::cout<<"Fail xnn_setup_convolution2d_nhwc_f32"<<std::endl;
            exit(1);
        }
    }

    void Compute(op_algo_runtime_context& rc) override{
        if(xnn_status_success != xnn_run_operator(maxpool_op, nullptr)){
            std::cout<<"Fail xnn_run_operator"<<std::endl;
            exit(1);
        }
    }

    ~MaxPool(){
        if(maxpool_op){
            if(xnn_status_success != xnn_delete_operator(maxpool_op)){
                std::cout<<"Fail xnn_delete_operator"<<std::endl;
                exit(1);
            }
            maxpool_op = nullptr;
        }
    }

private:
    Tensor x;
    Tensor y;
    std::vector<int32_t> strides;
    std::vector<int32_t> pads;
    std::vector<int32_t> kernel;
    xnn_operator_t maxpool_op = nullptr;
};

REGIST_OP_ALGO(MaxPool)
    .Device("CPU(XNNPACK)")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = MaxPool<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_xnnpack
} // end namespace mlfe
