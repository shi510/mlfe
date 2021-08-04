#include "mlfe/operators/maxpool2d.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/math/activations.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/device_context/cpu_context.h"
#include <cfloat>
#include <xnnpack.h>
#include <iostream>

namespace mlfe{
namespace operators{
namespace {

template <typename T>
struct maxpool2d_nhwc_op
{
    static void run(
                    Tensor x,
                    Tensor y,
                    std::vector<int32_t> psize,
                    std::vector<int32_t> strides
                    )
    {
        if(xnn_status_success != xnn_initialize(nullptr)){
            std::cout<<"Fail xnn_initialize"<<std::endl;
        }
        xnn_operator_t xnn_op = nullptr;
        int batch = x.shape()[0];
        int in_h = x.shape()[1];
        int in_w = x.shape()[2];
        int in_c = x.shape()[3];
        auto status = xnn_create_max_pooling2d_nhwc_f32(
            /*top=*/0, /*right=*/0,
            /*bottom=*/0, /*left=*/0,
            /*kernel_h=*/psize[0], /*kernel_h=*/psize[1],
            /*stride_h=*/strides[0], /*stride_w=*/strides[1],
            /*dilation height=*/1, /*dilation width=*/1,
            /*input_channels=*/in_c,
            /*input_pixel_stride=*/in_c,
            /*output_pixel_stride=*/in_c,
            /*output_min=*/-std::numeric_limits<T>::infinity(),
            /*output_max=*/std::numeric_limits<T>::infinity(),
            /*same_out_padding=*/0,
            &xnn_op);
        if(xnn_status_success != status){
            std::cout<<"Fail xnn_create_maxpooling2d_nhwc_f32"<<std::endl;
        }
        status = xnn_setup_max_pooling2d_nhwc_f32(
            xnn_op,
            batch, in_h, in_w,
            x.device_data<T>(),
            y.mutable_device_data<T>(),
            nullptr);
        if(xnn_status_success != status){
            std::cout<<"Fail xnn_setup_maxpooling2d_nhwc_f32"<<std::endl;
        }
        if(xnn_status_success != xnn_run_operator(xnn_op, nullptr)){
            std::cout<<"Fail xnn_run_operator: maxpooling2d_nhwc_f32"<<std::endl;
        }
        if(xnn_status_success != xnn_delete_operator(xnn_op)){
            std::cout<<"Fail xnn_delete_operator: maxpooling2d_nhwc_f32"<<std::endl;
        }
    }
};

template <class T>
struct maxpool2d_nhwc_grad_op
{
    static void run(
                    Tensor x,
                    Tensor y,
                    Tensor dy,
                    Tensor dx,
                    std::vector<int32_t> psize,
                    std::vector<int32_t> strides
                    )
    {
        int batch;
        int in_c, in_h, in_w;
        int out_h, out_w;
        batch = x.shape()[0];
        in_h = x.shape()[1];
        in_w = x.shape()[2];
        in_c = x.shape()[3];
        out_h = dy.shape()[1];
        out_w = dy.shape()[2];
        auto x_ptr = x.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();

        math::set<T, CPUContext>(dx.size(), T(0), dx_ptr);

        for(int n = 0; n < batch; ++n){
            for(int ph = 0; ph < out_h; ++ph){
                for(int pw = 0; pw < out_w; ++pw){
                    for(int c = 0; c < in_c; ++c){
                        int hstart = ph * strides[0];
                        int wstart = pw * strides[1];
                        int hend = std::min<int>(hstart + psize[0], in_h);
                        int wend = std::min<int>(wstart + psize[1], in_w);
                        const int pool_index = ph * out_w * in_c + pw * in_c + c;
                        int max_idx = -1;
                        T max_val = -FLT_MAX;
                        for(int h = hstart; h < hend; ++h){
                            for(int w = wstart; w < wend; ++w){
                                const int index = h * in_w * in_c + w * in_c + c;
                                if(x_ptr[index] > max_val){
                                    max_val = x_ptr[index];
                                    max_idx = index;
                                }
                            }
                        }
                        dx_ptr[max_idx] = dy_ptr[pool_index];
                    }
                }
            }
            x_ptr += in_h * in_w * in_c;
            dy_ptr += out_h * out_w * in_c;
            dx_ptr += in_h * in_w * in_c;
        }
    }
};

} // namespace anonymous

REGIST_OP_KERNEL(
    maxpool2d_fwd,
    maxpool2d_fwd_fn_t,
    maxpool2d_nhwc_op<float>::run
    );

REGIST_OP_KERNEL(
    maxpool2d_bwd,
    maxpool2d_bwd_fn_t,
    maxpool2d_nhwc_grad_op<float>::run
    );

} // namespace operators
} // namespace mlfe
