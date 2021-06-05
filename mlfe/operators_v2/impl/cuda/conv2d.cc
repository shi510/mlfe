#include "mlfe/operators_v2/conv2d.h"
#include "mlfe/operators_v2/impl/cuda/kernel/col2im.h"
#include "mlfe/operators_v2/impl/cuda/kernel/im2col.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/device_context/cuda_context.h"
#include "mlfe/math/blas.h"
#include <iostream>

namespace mlfe{
namespace operators_v2{
namespace {

template <typename T>
struct conv2d_nhwc_op
{
    static void run(
        Tensor x,
        Tensor kernel,
        Tensor y,
        std::vector<int32_t> strides,
        std::vector<int32_t> paddings
        )
    {
        std::vector<int32_t> filters_hw(2);
        filters_hw[0] = kernel.shape()[0];
        filters_hw[1] = kernel.shape()[1];
        // Output Filters.
        int m = kernel.shape()[3];
        // Output Feature Map Size.
        int n = y.shape()[1] * y.shape()[2];
        // Weight Size.
        int k = kernel.size() / m;
        int batch = x.shape()[0];
        int in_h = x.shape()[1];
        int in_w = x.shape()[2];
        int in_c = x.shape()[3];
        int out_h = y.shape()[1];
        int out_w = y.shape()[2];
        auto col_buf = create_memory(k * n * sizeof(T));
        auto x_ptr = x.device_data<T>();
        auto w_ptr = kernel.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        auto col_buf_ptr = col_buf->mutable_device_data<T>();
        CUDAContext c;
        for(int i = 0; i < batch; ++i){
            /*
            * image to column in range on kernel size.
            */
            cuda_kernel::im2col_nhwc<T>(
                in_c, in_h, in_w,
                out_h, out_w,
                filters_hw[0], filters_hw[1],
                strides[0], paddings[0],
                x_ptr, col_buf_ptr);

            /*
            * convolution with kernel.
            * kernel is learnable variable.
            * _w({_kernel_size, filters}) * x_col({_kernel_size, out_size})
            *  = _y({filters, out_size})
            */
            math::gemm<T, CUDAContext>(true, false,
                m, n, k,
                static_cast<T>(1), w_ptr, k,
                col_buf_ptr, n,
                static_cast<T>(0), y_ptr, n, &c);

            /*
            * next batch.
            */
            x_ptr += x.size() / batch;
            y_ptr += y.size() / batch;
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
        std::vector<int32_t> filters_hw(2);
        filters_hw[0] = kernel.shape()[0];
        filters_hw[1] = kernel.shape()[1];

        int32_t batch = dx.shape()[0];
        int32_t in_h = dx.shape()[1];
        int32_t in_w = dx.shape()[2];
        int32_t in_c = dx.shape()[3];
        int out_h = dy.shape()[1];
        int out_w = dy.shape()[2];

        // Output Filters.
        int32_t m = kernel.shape()[3];
        // Output Feature Map Size.
        int32_t n = dy.shape()[1] * dy.shape()[2];
        // Weight Size.
        int32_t k = kernel.size() / m;

        auto col_buf = create_memory(k * n * sizeof(T));
        auto w_ptr = kernel.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();
        auto col_ptr = col_buf->mutable_device_data<T>();

        CUDAContext c;
        for(int i = 0; i < batch; ++i){
            /*
            * Calculate loss to propagate through bottom.
            * w({kernel_size, filters}) * dy({filters, out_size})
            *  = col({kernel_size, out_size})
            */
            math::gemm<T, CUDAContext>(
                false, false, k, n, m,
                static_cast<T>(1), w_ptr, k,
                dy_ptr, n,
                static_cast<T>(0), col_ptr, n, &c
                );

            cuda_kernel::col2im_nhwc<T>(
                col_ptr,
                in_c, in_h, in_w,
                out_h, out_w,
                filters_hw[0], strides[0], paddings[0],
                dx_ptr
                );

            /*
            * next batch.
            */
            dx_ptr += dx.size() / batch;
            dy_ptr += dy.size() / batch;
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
        std::vector<int32_t> filters_hw(2);
        filters_hw[0] = dkernel.shape()[0];
        filters_hw[1] = dkernel.shape()[1];

        int32_t batch = x.shape()[0];
        int32_t in_h = x.shape()[1];
        int32_t in_w = x.shape()[2];
        int32_t in_c = x.shape()[3];
        int out_h = dy.shape()[1];
        int out_w = dy.shape()[2];

        // Output Filters.
        int32_t m = dkernel.shape()[3];
        // Output Feature Map Size.
        int32_t n = dy.shape()[1] * dy.shape()[2];
        // Weight Size.
        int32_t k = dkernel.size() / m;

        auto col_buf = create_memory(k * n * sizeof(T));
        auto x_ptr = x.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dw_ptr = dkernel.mutable_device_data<T>();
        auto col_ptr = col_buf->mutable_device_data<T>();
        CUDAContext c;
        for(int i = 0; i < batch; ++i){
            cuda_kernel::im2col_nhwc<T>(
                in_c, in_h, in_w,
                out_h, out_w,
                filters_hw[0], filters_hw[1],
                strides[0], paddings[0],
                x_ptr, col_ptr
                );

            /*
            * Calculate gradients of weights.
            * kernel_size ={kernel_h, kernel_w, channel_of_x} = k
            * filters ={number of feature map channel} = m
            * out_size ={y_h, y_w} = n
            * dy({filters, out_size}) * col({kernel_size, out_size})^T
            *  = dw({filters, kernel_size})
            */
            math::gemm<T, CUDAContext>(
                false, true, m, k, n,
                static_cast<T>(1), dy_ptr, n,
                col_ptr, n,
                static_cast<T>(1), dw_ptr, k, &c
                );

            /*
            * next batch.
            */
            x_ptr += x.size() / batch;
            dy_ptr += dy.size() / batch;
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
