#include "mlfe/operators_v2/conv2d.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/device_context/cuda_context.h"
#include "mlfe/operators/convolution_utils.h"
#include <cudnn.h>
#include <iostream>

namespace mlfe{
namespace operators_v2{
namespace {

void inline ASSERT_SUCCESS(cudnnStatus_t t){
    if(t != CUDNN_STATUS_SUCCESS){
        std::string e_msg;
        e_msg = "NOT CUDNN_STATUS_SUCCESS : ";
        e_msg += cudnnGetErrorString(t);
        std::cout<<"Fail CUDNN_STATUS_SUCCESS"<<std::endl;
        throw e_msg;
    }
}

template <typename T>
auto get_cudnn_type() {
    if(std::is_same<T, float>::value){
        return CUDNN_DATA_FLOAT;
    }
    else if(std::is_same<T, double>::value){
        return CUDNN_DATA_DOUBLE;
    }
    return CUDNN_DATA_FLOAT;
};

struct cudnn_conv2d_arguments{
    void *ws_fwd;
    size_t ws_fwd_size;
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t x_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnTensorDescriptor_t y_desc;
    cudnnConvolutionFwdAlgo_t conv_algo;
    cudnnConvolutionDescriptor_t conv_desc;
};

template <typename T>
void create_cudnn_args(
    cudnn_conv2d_arguments * args,
    Tensor &x,
    Tensor &y,
    Tensor &kernel,
    std::vector<int32_t> &strides,
    std::vector<int32_t> &paddings
    )
{
    cudnnCreate(&args->handle);
    cudnnCreateTensorDescriptor(&args->x_desc);
    cudnnCreateFilterDescriptor(&args->w_desc);
    cudnnCreateTensorDescriptor(&args->y_desc);
    cudnnCreateConvolutionDescriptor(&args->conv_desc);

    cudnnSetTensor4dDescriptor(
        args->x_desc,
        CUDNN_TENSOR_NHWC,
        get_cudnn_type<T>(),
        x.shape()[0], x.shape()[3], x.shape()[1], x.shape()[2]);

    cudnnSetFilter4dDescriptor(
        args->w_desc,
        get_cudnn_type<T>(),
        CUDNN_TENSOR_NHWC,
        kernel.shape()[3], kernel.shape()[2], kernel.shape()[1], kernel.shape()[0]);

    cudnnSetTensor4dDescriptor(
        args->y_desc,
        CUDNN_TENSOR_NHWC,
        get_cudnn_type<T>(),
        y.shape()[0], y.shape()[3], y.shape()[1], y.shape()[2]);

    ASSERT_SUCCESS(cudnnSetConvolution2dDescriptor(
        args->conv_desc,
        paddings[0], paddings[1],
        strides[0], strides[1], 1, 1,
        CUDNN_CROSS_CORRELATION,
        get_cudnn_type<T>()
    ));

    ASSERT_SUCCESS(cudnnGetConvolutionForwardAlgorithm(
        args->handle,
        args->x_desc,
        args->w_desc,
        args->conv_desc,
        args->y_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &args->conv_algo
    ));

    ASSERT_SUCCESS(cudnnGetConvolutionForwardWorkspaceSize(
        args->handle,
        args->x_desc,
        args->w_desc,
        args->conv_desc,
        args->y_desc,
        args->conv_algo,
        &args->ws_fwd_size
    ));
    cudaMalloc(&args->ws_fwd, args->ws_fwd_size);
}

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
        // int64_t key =
        //     kernel.size() + strides[0] + strides[1] + paddings[0] + paddings[1];
        // auto args_ptr = get_object_pool().search<cudnn_conv2d_arguments>(key);
        // if(!args_ptr){
        //     auto new_args = std::make_shared<cudnn_conv2d_arguments>();
        //     get_object_pool().regist<cudnn_conv2d_arguments>(new_args);
        //     create_cudnn_args(new_args.get(), x, kernel, y, strides, paddings);
        //     args_ptr = new_args;
        // }

        void *_ws_fwd;
        size_t _ws_fwd_size;
        cudnnHandle_t _handle;
        cudnnTensorDescriptor_t _x_desc;
        cudnnFilterDescriptor_t _w_desc;
        cudnnTensorDescriptor_t _y_desc;
        cudnnConvolutionFwdAlgo_t _conv_algo;
        cudnnConvolutionDescriptor_t _conv_desc;

        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_x_desc);
        cudnnCreateFilterDescriptor(&_w_desc);
        cudnnCreateTensorDescriptor(&_y_desc);
        cudnnCreateConvolutionDescriptor(&_conv_desc);

        cudnnSetTensor4dDescriptor(
            _x_desc,
            CUDNN_TENSOR_NHWC,
            get_cudnn_type<T>(),
            x.shape()[0], x.shape()[3], x.shape()[1], x.shape()[2]);

        cudnnSetFilter4dDescriptor(
            _w_desc,
            get_cudnn_type<T>(),
            CUDNN_TENSOR_NHWC,
            kernel.shape()[3], kernel.shape()[2], kernel.shape()[1], kernel.shape()[0]);

        cudnnSetTensor4dDescriptor(
            _y_desc,
            CUDNN_TENSOR_NHWC,
            get_cudnn_type<T>(),
            y.shape()[0], y.shape()[3], y.shape()[1], y.shape()[2]);

        ASSERT_SUCCESS(cudnnSetConvolution2dDescriptor(
            _conv_desc,
            paddings[0], paddings[1],
            strides[0], strides[1], 1, 1,
            CUDNN_CROSS_CORRELATION,
            get_cudnn_type<T>()
        ));

        ASSERT_SUCCESS(cudnnGetConvolutionForwardAlgorithm(
            _handle,
            _x_desc,
            _w_desc,
            _conv_desc,
            _y_desc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &_conv_algo
        ));

        ASSERT_SUCCESS(cudnnGetConvolutionForwardWorkspaceSize(
            _handle,
            _x_desc,
            _w_desc,
            _conv_desc,
            _y_desc,
            _conv_algo,
            &_ws_fwd_size
        ));

        cudaMalloc(&_ws_fwd, _ws_fwd_size);

        const float alpha = 1, beta = 0;
        ASSERT_SUCCESS(cudnnConvolutionForward(
            _handle,
            &alpha,
            _x_desc, x.device_data<void>(),
            _w_desc, kernel.device_data<void>(),
            _conv_desc, _conv_algo,
            _ws_fwd, _ws_fwd_size,
            &beta,
            _y_desc, y.mutable_device_data<void>()
        ));

        cudnnDestroy(_handle);
        cudnnDestroyTensorDescriptor(_x_desc);
        cudnnDestroyFilterDescriptor(_w_desc);
        cudnnDestroyTensorDescriptor(_y_desc);
        cudnnDestroyConvolutionDescriptor(_conv_desc);
        cudaFree(_ws_fwd);
        _ws_fwd = nullptr;
        _ws_fwd_size = 0;
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
        void *_ws_data;
        size_t _ws_data_size;
        cudnnHandle_t _handle;
        cudnnConvolutionFwdAlgo_t _conv_algo;
        cudnnConvolutionBwdDataAlgo_t _data_algo;
        cudnnTensorDescriptor_t _dx_desc;
        cudnnFilterDescriptor_t _w_desc;
        cudnnTensorDescriptor_t _dy_desc;
        cudnnConvolutionDescriptor_t _conv_desc;

        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_dx_desc);
        cudnnCreateFilterDescriptor(&_w_desc);
        cudnnCreateTensorDescriptor(&_dy_desc);
        cudnnCreateConvolutionDescriptor(&_conv_desc);

        cudnnSetFilter4dDescriptor(_w_desc,
            get_cudnn_type<T>(),
            CUDNN_TENSOR_NHWC,
            kernel.shape()[3], kernel.shape()[2], kernel.shape()[1], kernel.shape()[0]);

        cudnnSetTensor4dDescriptor(_dy_desc, CUDNN_TENSOR_NHWC,
            get_cudnn_type<T>(),
            dy.shape()[0], dy.shape()[3], dy.shape()[1], dy.shape()[2]);

        cudnnSetTensor4dDescriptor(_dx_desc, CUDNN_TENSOR_NHWC,
            get_cudnn_type<T>(),
            dx.shape()[0], dx.shape()[3], dx.shape()[1], dx.shape()[2]);

        cudnnSetConvolution2dDescriptor(
            _conv_desc,
            paddings[0], paddings[1],
            strides[0], strides[1], 1, 1,
            CUDNN_CROSS_CORRELATION,
            get_cudnn_type<T>()
        );

        cudnnGetConvolutionBackwardDataAlgorithm(
            _handle,
            _w_desc,
            _dy_desc,
            _conv_desc,
            _dx_desc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &_data_algo
        );

        cudnnGetConvolutionBackwardDataWorkspaceSize(
            _handle,
            _w_desc,
            _dy_desc,
            _conv_desc,
            _dx_desc,
            _data_algo,
            &_ws_data_size
        );
        cudaMalloc(&_ws_data, _ws_data_size);
        const float alpha = 1, beta = 0;

        ASSERT_SUCCESS(cudnnConvolutionBackwardData(
            _handle,
            &alpha,
            _w_desc, kernel.device_data<void>(),
            _dy_desc, dy.device_data<void>(),
            _conv_desc, _data_algo,
            _ws_data, _ws_data_size,
            &beta,
            _dx_desc, dx.mutable_device_data<void>()
        ));

        cudnnDestroyTensorDescriptor(_dx_desc);
        cudnnDestroyFilterDescriptor(_w_desc);
        cudnnDestroyTensorDescriptor(_dy_desc);
        cudnnDestroyConvolutionDescriptor(_conv_desc);
        cudaFree(_ws_data);
        cudnnDestroy(_handle);
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
        void *_ws_filter;
        size_t _ws_filter_size;
        cudnnHandle_t _handle;
        cudnnConvolutionFwdAlgo_t _conv_algo;
        cudnnConvolutionBwdFilterAlgo_t _filter_algo;
        cudnnTensorDescriptor_t _x_desc;
        cudnnFilterDescriptor_t _dw_desc;
        cudnnTensorDescriptor_t _dy_desc;
        cudnnConvolutionDescriptor_t _conv_desc;

        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_x_desc);
        cudnnCreateFilterDescriptor(&_dw_desc);
        cudnnCreateTensorDescriptor(&_dy_desc);
        cudnnCreateConvolutionDescriptor(&_conv_desc);

        cudnnSetTensor4dDescriptor(_x_desc, CUDNN_TENSOR_NHWC,
            get_cudnn_type<T>(),
            x.shape()[0], x.shape()[3], x.shape()[1], x.shape()[2]);

        cudnnSetFilter4dDescriptor(_dw_desc,
            get_cudnn_type<T>(),
            CUDNN_TENSOR_NHWC,
            dkernel.shape()[3], dkernel.shape()[2], dkernel.shape()[1], dkernel.shape()[0]);

        cudnnSetTensor4dDescriptor(_dy_desc, CUDNN_TENSOR_NHWC,
            get_cudnn_type<T>(),
            dy.shape()[0], dy.shape()[3], dy.shape()[1], dy.shape()[2]);

        cudnnSetConvolution2dDescriptor(
            _conv_desc,
            paddings[0], paddings[1],
            strides[0], strides[1], 1, 1,
            CUDNN_CROSS_CORRELATION,
            get_cudnn_type<T>()
        );

        cudnnGetConvolutionForwardAlgorithm(
            _handle,
            _x_desc,
            _dw_desc,
            _conv_desc,
            _dy_desc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &_conv_algo
        );

        cudnnGetConvolutionBackwardFilterAlgorithm(
            _handle,
            _x_desc,
            _dy_desc,
            _conv_desc,
            _dw_desc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &_filter_algo
        );

        cudnnGetConvolutionBackwardFilterWorkspaceSize(
            _handle,
            _x_desc,
            _dy_desc,
            _conv_desc,
            _dw_desc,
            _filter_algo,
            &_ws_filter_size
        );

        cudaMalloc(&_ws_filter, _ws_filter_size);

        const float alpha = 1, beta = 0;

        ASSERT_SUCCESS(cudnnConvolutionBackwardFilter(
            _handle,
            &alpha, _x_desc, x.device_data<void>(),
            _dy_desc, dy.device_data<void>(),
            _conv_desc, _filter_algo,
            _ws_filter, _ws_filter_size,
            &beta,
            _dw_desc, dkernel.mutable_device_data<void>()
        ));

        cudnnDestroyTensorDescriptor(_x_desc);
        cudnnDestroyFilterDescriptor(_dw_desc);
        cudnnDestroyTensorDescriptor(_dy_desc);
        cudnnDestroyConvolutionDescriptor(_conv_desc);
        cudaFree(_ws_filter);
        cudnnDestroy(_handle);
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
