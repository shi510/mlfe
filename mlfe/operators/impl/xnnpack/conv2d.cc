#include "mlfe/operators/conv2d.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/device_context/cpu_context.h"
#include "mlfe/core/device.h"
#include "mlfe/math/blas.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/math/transform.h"
#include "mlfe/device_context/cpu_context.h"
#include "mlfe/operators/utils.h"
#include <xnnpack.h>
#include <iostream>

namespace mlfe{
namespace operators{
namespace {

template <typename T>
struct conv2d_nhwc_op
{
    static void run(Tensor x,
        Tensor kernel,
        Tensor y,
        std::vector<int32_t> strides,
        std::vector<int32_t> paddings
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
        int kh = kernel.shape()[0];
        int kw = kernel.shape()[1];
        auto status = xnn_create_convolution2d_nhwc_f32(
            paddings[0], paddings[1], paddings[0], paddings[1],
            kh, kw,
            /*subsampling_height=*/strides[0],
            /*subsampling_width=*/strides[1],
            /*dilation height=*/1,
            /*dilation width=*/1,
            /*groups=*/1,
            /*group_input_channels=*/in_c,
            /*group_output_channels=*/y.shape()[3],
            /*input_pixel_stride=*/in_c,
            /*output_pixel_stride=*/y.shape()[3],
            /*weight_ptr=*/kernel.device_data<T>(),
            /*bias_ptr=*/nullptr,
            /*output_min=*/-std::numeric_limits<T>::infinity(),
            /*output_max=*/std::numeric_limits<T>::infinity(),
            /*depthwise_layout=*/0,
            &xnn_op);
        if(xnn_status_success != status){
            std::cout<<"Fail xnn_create_convolution2d_nhwc_f32"<<std::endl;
        }
        status = xnn_setup_convolution2d_nhwc_f32(
            xnn_op,
            batch, in_h, in_w,
            x.device_data<T>(), y.mutable_device_data<T>(),
            nullptr);
        if(xnn_status_success != status){
            std::cout<<"Fail xnn_setup_convolution2d_nhwc_f32"<<std::endl;
        }
        if(xnn_status_success != xnn_run_operator(xnn_op, nullptr)){
            std::cout<<"Fail xnn_run_operator"<<std::endl;
        }
        if(xnn_status_success != xnn_delete_operator(xnn_op)){
            std::cout<<"Fail xnn_delete_operator"<<std::endl;
        }
    }
};

template <typename T>
struct conv2d_nhwc_input_grad_op
{
    static void run(
        Tensor kernel,
        Tensor dy,
        Tensor dx,
        std::vector<int32_t> strides,
        std::vector<int32_t> paddings
        )
    {
        using IntVec = std::vector<type::int32::T>;
        memory_ptr col_buf;
        int m, n, k, batch;
        int in_c, in_h, in_w;
        std::vector<type::int32::T> filters_hw;
        filters_hw.resize(2);
        filters_hw[0] = kernel.shape()[0];
        filters_hw[1] = kernel.shape()[1];

        batch = dx.shape()[0];
        in_h = dx.shape()[1];
        in_w = dx.shape()[2];
        in_c = dx.shape()[3];

        // output channels.
        m = kernel.shape()[3];
        // output height * output width
        n = dy.shape()[1] * dy.shape()[2];
        // input channels * kernel height * kernel width
        k = kernel.shape()[2] * filters_hw[0] * filters_hw[1];

        col_buf = create_memory(k * n * sizeof(T));

        auto w_ptr = kernel.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();
        auto col_ptr = col_buf->mutable_device_data<T>();

        math::set<T, CPUContext>(dx.size(), static_cast<T>(0), dx_ptr);

        for(int i = 0; i < batch; ++i){
            /*
            * Calculate loss to propagate through bottom.
            * w({kernel_size, out_channel}) * dy({out_size, out_channel})^T
            *  = col({kernel_size, out_size})
            */
            math::gemm<T, CPUContext>(
                false, true, k, n, m,
                static_cast<T>(1), w_ptr, k,
                dy_ptr, n,
                static_cast<T>(0), col_ptr, n, nullptr
                );

            math::col2im_nhwc<T, CPUContext>(
                col_ptr,
                in_c, in_h, in_w,
                filters_hw[0], strides[0], paddings[0],
                dx_ptr
                );

            /*
            * next batch.
            */
            dx_ptr += dx.size() / batch;
            dy_ptr += n * m;
        }
    }

};

template <typename T>
struct conv2d_nhwc_kernel_grad_op{

    static void run(
        Tensor x,
        Tensor dy,
        Tensor dkernel,
        std::vector<int32_t> strides,
        std::vector<int32_t> paddings
        )
    {
        using IntVec = std::vector<type::int32::T>;
        memory_ptr col_buf;
        int m, n, k, batch;
        int in_c, in_h, in_w;
        std::vector<type::int32::T> kernel_hw =
            {dkernel.shape()[0], dkernel.shape()[1]};
        batch = x.shape()[0];
        in_h = x.shape()[1];
        in_w = x.shape()[2];
        in_c = x.shape()[3];

        // output channels.
        m = dkernel.shape()[3];
        // output height * width
        n = dy.shape()[1] * dy.shape()[2];
        // in_channels * kernel_height * kernel_width
        k = x.shape()[3] * kernel_hw[0] * kernel_hw[1];

        col_buf = create_memory(k * n * sizeof(T));

        auto x_ptr = x.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dw_ptr = dkernel.mutable_device_data<T>();
        auto col_ptr = col_buf->mutable_device_data<T>();

        math::set<T, CPUContext>(dkernel.size(), static_cast<T>(0), dw_ptr);
        math::set<T, CPUContext>(k * n, static_cast<T>(0), col_ptr);

        for(int i = 0; i < batch; ++i){
            math::im2col_nhwc<T, CPUContext>(
                in_c, in_h, in_w,
                kernel_hw[0], kernel_hw[1],
                strides[0], paddings[0],
                x_ptr, col_ptr
                );

            /*
            * Calculate gradients of weights.
            * kernel_size ={kernel_h, kernel_w, channel_of_x} = k
            * filters ={number of feature map channel} = m
            * out_size ={y_h, y_w} = n
            * col({kernel_size, out_size}) * dy({filters, out_size})^T
            *  = dw({filters, kernel_size})
            */
            math::gemm<T, CPUContext>(
                false, true, k, m, n,
                static_cast<T>(1), col_ptr, n,
                dy_ptr, n,
                static_cast<T>(1), dw_ptr, k, nullptr
                );

            /*
            * next batch.
            */
            x_ptr += x.size() / batch;
            dy_ptr += n * m;
        }
    }
};

} // namespace anonymous

REGIST_OP_KERNEL(
    conv2d_fwd,
    conv2d_fwd_fn_t,
    conv2d_nhwc_op<float>::run
    );

REGIST_OP_KERNEL(
    conv2d_input_bwd,
    conv2d_input_bwd_fn_t,
    conv2d_nhwc_input_grad_op<float>::run
    );

REGIST_OP_KERNEL(
    conv2d_kernel_bwd,
    conv2d_kernel_bwd_fn_t,
    conv2d_nhwc_kernel_grad_op<float>::run
    );

} // namespace operators
} // namespace mlfe
