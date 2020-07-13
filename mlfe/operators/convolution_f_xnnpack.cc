#include "mlfe/core/op_algo.h"
#include "mlfe/core/device.h"
#include "mlfe/operators/convolution_utils.h"
#include <memory>
#include <limits>
#include <iostream>
#include <xnnpack.h>

namespace mlfe{
namespace algorithm_xnnpack{

/*
 * Data order: NHWC
 * Input: X, W, B (optional)
 * Output: Y
 * W's Shape: [X's Channel, kernel's Height, kernel's Width, Y's Channel]
 * B's Shape: [Y's Channel]
 */
template <class Tp>
class Convolution : public OpAlgo{
using T = typename Tp::T;
using IntVec = std::vector<type::int32::T>;
public:
    Convolution(OpAlgoContext *oac) : OpAlgo(oac, "Convolution"){
        if(xnn_status_success != xnn_initialize(nullptr)){
            std::cout<<"Fail xnn_initialize"<<std::endl;
            exit(1);
        }
        y = oac->get_output(0);
        x = oac->get_input(0);
        w = oac->get_input(1);
        strides = oac->get_attr<IntVec>("strides");
        if(oac->has_attr("same_out")){
            auto pad_h = util::calc_conv2d_pad_size_for_same_output(
                x.shape()[1], w.shape()[1], strides[0]
            );
            auto pad_w = util::calc_conv2d_pad_size_for_same_output(
                x.shape()[2], w.shape()[2], strides[1]
            );
            pads.push_back(pad_h);
            pads.push_back(pad_w);
        }
        else{
            pads = oac->get_attr<IntVec>("pads");
        }
        resize();
    }

    void resize() override{
        batch = x.shape()[0];
        in_h = x.shape()[1];
        in_w = x.shape()[2];
        in_c = x.shape()[3];
        kh = w.shape()[1];
        kw = w.shape()[2];
        int out_h = util::calc_conv2d_output(
            in_h, kh, strides[0], pads[0]
        );
        int out_w = util::calc_conv2d_output(
            in_w, kw, strides[1], pads[1]
        );
        y.resize({x.shape()[0], out_h, out_w, w.shape()[0]});
        auto status = xnn_create_convolution2d_nhwc_f32(
            pads[0], pads[1], pads[0], pads[1],
            kh, kw,
            /*subsampling_height=*/strides[0], /*subsampling_width=*/strides[1],
            /*dilation height=*/1, /*dilation width=*/1,
            /*groups=*/1,
            /*group_input_channels=*/in_c,
            /*group_output_channels=*/y.shape()[3],
            /*input_pixel_stride=*/in_c, /*output_pixel_stride=*/y.shape()[3],
            w.device_data<float>(), /*bias ptr=*/nullptr,
            /*output_min=*/std::numeric_limits<float>::min(),
            /*output_max=*/std::numeric_limits<float>::max(),
            /*depthwise_layout=*/0,
            &convolution_op);
        if(xnn_status_success != status){
            std::cout<<"Fail xnn_create_convolution2d_nhwc_f32"<<std::endl;
            exit(1);
        }
        status = xnn_setup_convolution2d_nhwc_f32(
            convolution_op,
            batch, in_h, in_w,
            x.device_data<float>(), y.mutable_device_data<float>(),
            nullptr);
        if(xnn_status_success != status){
            std::cout<<"Fail xnn_setup_convolution2d_nhwc_f32"<<std::endl;
            exit(1);
        }
    }

    void Compute(op_algo_runtime_context& rc) override{
        if(xnn_status_success != xnn_run_operator(convolution_op, nullptr)){
            std::cout<<"Fail xnn_run_operator"<<std::endl;
            exit(1);
        }
    }

    ~Convolution(){
        if(convolution_op){
            if(xnn_status_success != xnn_delete_operator(convolution_op)){
                std::cout<<"Fail xnn_delete_operator"<<std::endl;
                exit(1);
            }
            convolution_op = nullptr;
        }
    }

private:
    Tensor x;
    Tensor w;
    Tensor y;
    int batch;
    int in_h;
    int in_w;
    int in_c;
    int out_c;
    int kh;
    int kw;
    std::vector<int32_t> strides;
    std::vector<int32_t> pads;
    xnn_operator_t convolution_op = nullptr;
};

REGIST_OP_ALGO(Convolution)
    .Device("CPU(XNNPACK)")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Convolution<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

/*
 * Data order: NHWC
 * Input: X, W, B (optional)
 * Output: Y
 * W's Shape: [X's Channel, kernel's Height, kernel's Width, Y's Channel]
 * B's Shape: [Y's Channel]
 */
template <class Tp>
class DepthwiseConv2d : public OpAlgo{
using T = typename Tp::T;
using IntVec = std::vector<type::int32::T>;
public:
    DepthwiseConv2d(OpAlgoContext *oac) : OpAlgo(oac, "DepthwiseConv2d"){
        if(xnn_status_success != xnn_initialize(nullptr)){
            std::cout<<"Fail xnn_initialize"<<std::endl;
            exit(1);
        }
        y = oac->get_output(0);
        x = oac->get_input(0);
        w = oac->get_input(1);
        strides = oac->get_attr<IntVec>("strides");
        pads = oac->get_attr<IntVec>("pads");
        resize();
    }

    void resize() override{
        int batch = x.shape()[0];
        int in_h = x.shape()[1];
        int in_w = x.shape()[2];
        int in_c = x.shape()[3];
        int out_h = util::calc_conv2d_output(
            in_h, w.shape()[0], strides[0], pads[0]
        );
        int out_w = util::calc_conv2d_output(
            in_w, w.shape()[1], strides[1], pads[1]
        );
        y.resize({x.shape()[0], out_h, out_w, in_c});
        auto status = xnn_create_convolution2d_nhwc_f32(
            pads[0], pads[1], pads[0], pads[1],
            w.shape()[0], w.shape()[1],
            /*subsampling_height=*/strides[0], /*subsampling_width=*/strides[1],
            /*dilation height=*/1, /*dilation width=*/1,
            /*groups=*/in_c,
            /*group_input_channels=*/1,
            /*group_output_channels=*/1,
            /*input_pixel_stride=*/in_c, /*output_pixel_stride=*/in_c,
            w.device_data<float>(), /*bias ptr=*/nullptr,
            /*output_min=*/std::numeric_limits<float>::min(),
            /*output_max=*/std::numeric_limits<float>::max(),
            /*flags=*/XNN_FLAG_DEPTHWISE_CONVOLUTION,
            &convolution_op);
        if(xnn_status_success != status){
            std::cout<<"Fail xnn_create_convolution2d_nhwc_f32"<<std::endl;
            exit(1);
        }
        status = xnn_setup_convolution2d_nhwc_f32(
            convolution_op,
            batch, in_h, in_w,
            x.device_data<float>(), y.mutable_device_data<float>(),
            nullptr);
        if(xnn_status_success != status){
            std::cout<<"Fail xnn_setup_convolution2d_nhwc_f32"<<std::endl;
            exit(1);
        }
    }

    void Compute(op_algo_runtime_context& rc) override{
        if(xnn_status_success != xnn_run_operator(convolution_op, nullptr)){
            std::cout<<"Fail xnn_run_operator"<<std::endl;
            exit(1);
        }
    }

    ~DepthwiseConv2d(){
        if(convolution_op){
            if(xnn_status_success != xnn_delete_operator(convolution_op)){
                std::cout<<"Fail xnn_delete_operator"<<std::endl;
                exit(1);
            }
            convolution_op = nullptr;
        }
    }

private:
    Tensor x;
    Tensor w;
    Tensor y;
    std::vector<int32_t> strides;
    std::vector<int32_t> pads;
    xnn_operator_t convolution_op = nullptr;
};

REGIST_OP_ALGO(DepthwiseConv2d)
    .Device("CPU(XNNPACK)")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = DepthwiseConv2d<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_xnnpack
} // end namespace mlfe
