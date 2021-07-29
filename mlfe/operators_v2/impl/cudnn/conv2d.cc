#include "mlfe/operators/conv2d.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/device_context/cuda_context.h"
#include <cudnn.h>
#include <iostream>
#include <unordered_map>

namespace mlfe{
namespace operators{
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

struct cudnn_conv2d_fwd_arguments{
    void free(){
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyFilterDescriptor(w_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudaFree(ws_fwd);
        ws_fwd = nullptr;
        ws_fwd_size = 0;
    }
    void *ws_fwd;
    size_t ws_fwd_size;
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t x_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnTensorDescriptor_t y_desc;
    cudnnConvolutionFwdAlgo_t conv_algo;
    cudnnConvolutionDescriptor_t conv_desc;
};

struct cudnn_conv2d_bwd_input_arguments{
    void free(){
        cudnnDestroyTensorDescriptor(_dx_desc);
        cudnnDestroyFilterDescriptor(_w_desc);
        cudnnDestroyTensorDescriptor(_dy_desc);
        cudnnDestroyConvolutionDescriptor(_conv_desc);
        cudaFree(_ws_data);
    }
    void *_ws_data;
    size_t _ws_data_size;
    cudnnHandle_t _handle;
    cudnnConvolutionFwdAlgo_t _conv_algo;
    cudnnConvolutionBwdDataAlgo_t _data_algo;
    cudnnTensorDescriptor_t _dx_desc;
    cudnnFilterDescriptor_t _w_desc;
    cudnnTensorDescriptor_t _dy_desc;
    cudnnConvolutionDescriptor_t _conv_desc;
};

struct cudnn_conv2d_bwd_kernel_arguments{
    void free(){
        cudnnDestroyTensorDescriptor(_x_desc);
        cudnnDestroyFilterDescriptor(_dw_desc);
        cudnnDestroyTensorDescriptor(_dy_desc);
        cudnnDestroyConvolutionDescriptor(_conv_desc);
        cudaFree(_ws_filter);
    }
    void *_ws_filter;
    size_t _ws_filter_size;
    cudnnHandle_t _handle;
    cudnnConvolutionFwdAlgo_t _conv_algo;
    cudnnConvolutionBwdFilterAlgo_t _filter_algo;
    cudnnTensorDescriptor_t _x_desc;
    cudnnFilterDescriptor_t _dw_desc;
    cudnnTensorDescriptor_t _dy_desc;
    cudnnConvolutionDescriptor_t _conv_desc;
};

template <typename T>
void create_cudnn_args(
    cudnn_conv2d_fwd_arguments * args,
    Tensor x,
    Tensor kernel,
    Tensor y,
    std::vector<int32_t> strides,
    std::vector<int32_t> paddings
    )
{
    args->handle = cuda_context_v2::create()->get_cudnn_handle();
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
void create_cudnn_args_bwd_input(
    cudnn_conv2d_bwd_input_arguments * args,
    Tensor kernel,
    Tensor dy,
    Tensor dx,
    std::vector<int32_t> strides,
    std::vector<int32_t> paddings
    )
{
    args->_handle = cuda_context_v2::create()->get_cudnn_handle();
    cudnnCreateTensorDescriptor(&args->_dx_desc);
    cudnnCreateFilterDescriptor(&args->_w_desc);
    cudnnCreateTensorDescriptor(&args->_dy_desc);
    cudnnCreateConvolutionDescriptor(&args->_conv_desc);

    cudnnSetFilter4dDescriptor(args->_w_desc,
        get_cudnn_type<T>(),
        CUDNN_TENSOR_NHWC,
        kernel.shape()[3], kernel.shape()[2], kernel.shape()[1], kernel.shape()[0]);

    cudnnSetTensor4dDescriptor(args->_dy_desc, CUDNN_TENSOR_NHWC,
        get_cudnn_type<T>(),
        dy.shape()[0], dy.shape()[3], dy.shape()[1], dy.shape()[2]);

    cudnnSetTensor4dDescriptor(args->_dx_desc, CUDNN_TENSOR_NHWC,
        get_cudnn_type<T>(),
        dx.shape()[0], dx.shape()[3], dx.shape()[1], dx.shape()[2]);

    cudnnSetConvolution2dDescriptor(
        args->_conv_desc,
        paddings[0], paddings[1],
        strides[0], strides[1], 1, 1,
        CUDNN_CROSS_CORRELATION,
        get_cudnn_type<T>()
    );

    cudnnGetConvolutionBackwardDataAlgorithm(
        args->_handle,
        args->_w_desc,
        args->_dy_desc,
        args->_conv_desc,
        args->_dx_desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &args->_data_algo
    );

    cudnnGetConvolutionBackwardDataWorkspaceSize(
        args->_handle,
        args->_w_desc,
        args->_dy_desc,
        args->_conv_desc,
        args->_dx_desc,
        args->_data_algo,
        &args->_ws_data_size
    );
    cudaMalloc(&args->_ws_data, args->_ws_data_size);
}

template <typename T>
void create_cudnn_args_bwd_kernel(
    cudnn_conv2d_bwd_kernel_arguments * args,
    Tensor x,
    Tensor dy,
    Tensor dkernel,
    std::vector<int32_t> strides,
    std::vector<int32_t> paddings
    )
{
    args->_handle = cuda_context_v2::create()->get_cudnn_handle();
    cudnnCreateTensorDescriptor(&args->_x_desc);
    cudnnCreateFilterDescriptor(&args->_dw_desc);
    cudnnCreateTensorDescriptor(&args->_dy_desc);
    cudnnCreateConvolutionDescriptor(&args->_conv_desc);

    cudnnSetTensor4dDescriptor(args->_x_desc, CUDNN_TENSOR_NHWC,
        get_cudnn_type<T>(),
        x.shape()[0], x.shape()[3], x.shape()[1], x.shape()[2]);

    cudnnSetFilter4dDescriptor(args->_dw_desc,
        get_cudnn_type<T>(),
        CUDNN_TENSOR_NHWC,
        dkernel.shape()[3], dkernel.shape()[2], dkernel.shape()[1], dkernel.shape()[0]);

    cudnnSetTensor4dDescriptor(args->_dy_desc, CUDNN_TENSOR_NHWC,
        get_cudnn_type<T>(),
        dy.shape()[0], dy.shape()[3], dy.shape()[1], dy.shape()[2]);

    cudnnSetConvolution2dDescriptor(
        args->_conv_desc,
        paddings[0], paddings[1],
        strides[0], strides[1], 1, 1,
        CUDNN_CROSS_CORRELATION,
        get_cudnn_type<T>()
    );

    cudnnGetConvolutionForwardAlgorithm(
        args->_handle,
        args->_x_desc,
        args->_dw_desc,
        args->_conv_desc,
        args->_dy_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &args->_conv_algo
    );

    cudnnGetConvolutionBackwardFilterAlgorithm(
        args->_handle,
        args->_x_desc,
        args->_dy_desc,
        args->_conv_desc,
        args->_dw_desc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        0,
        &args->_filter_algo
    );

    cudnnGetConvolutionBackwardFilterWorkspaceSize(
        args->_handle,
        args->_x_desc,
        args->_dy_desc,
        args->_conv_desc,
        args->_dw_desc,
        args->_filter_algo,
        &args->_ws_filter_size
    );

    cudaMalloc(&args->_ws_filter, args->_ws_filter_size);
}

struct object_collector{

    template <typename T>
    T * search(std::string key)
    {
        auto it = map.find(key);
        if(it == map.end()){
            return nullptr;
        }
        return static_cast<T *>(it->second);
    }

    template <typename T>
    void regist(std::string key, T obj)
    {
        T * obj_ptr = new T;
        *obj_ptr = obj;
        map[key] = static_cast<void *>(obj_ptr);
    }

    std::unordered_map<std::string, void *> map;
};

object_collector * get_object_pool()
{
    static object_collector pool = object_collector();
    return &pool;
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
        // std::string key =
        //     "conv2d_fwd/" + 
        //     std::to_string(x.get_node().get_order()) +
        //     std::to_string(kernel.get_node().get_order()) +
        //     std::to_string(kernel.size()) +
        //     std::to_string(strides[0]) +
        //     std::to_string(strides[1]) +
        //     std::to_string(paddings[0]) +
        //     std::to_string(paddings[1]);
        // auto args_ptr = get_object_pool()->search<cudnn_conv2d_fwd_arguments>(key);
        // if(!args_ptr){
        //     cudnn_conv2d_fwd_arguments new_args;
        //     create_cudnn_args<T>(&new_args, x, kernel, y, strides, paddings);
        //     get_object_pool()->regist(key, new_args);
        //     args_ptr = get_object_pool()->search<cudnn_conv2d_fwd_arguments>(key);
        // }
        cudnn_conv2d_fwd_arguments new_args;
        create_cudnn_args<T>(&new_args, x, kernel, y, strides, paddings);
        cudnn_conv2d_fwd_arguments *args_ptr = &new_args;

        const float alpha = 1, beta = 0;
        ASSERT_SUCCESS(cudnnConvolutionForward(
            args_ptr->handle,
            &alpha,
            args_ptr->x_desc, x.device_data<void>(),
            args_ptr->w_desc, kernel.device_data<void>(),
            args_ptr->conv_desc, args_ptr->conv_algo,
            args_ptr->ws_fwd, args_ptr->ws_fwd_size,
            &beta,
            args_ptr->y_desc, y.mutable_device_data<void>()
        ));
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
        // std::string key =
        //     "conv2d_bwd_input/" + 
        //     // std::to_string(dy.get_node().get_order()) +
        //     std::to_string(kernel.get_node().get_order()) +
        //     std::to_string(kernel.size()) +
        //     std::to_string(strides[0]) +
        //     std::to_string(strides[1]) +
        //     std::to_string(paddings[0]) +
        //     std::to_string(paddings[1]);
        // auto args_ptr = get_object_pool()->search<cudnn_conv2d_bwd_input_arguments>(key);
        // if(!args_ptr){
        //     cudnn_conv2d_bwd_input_arguments new_args;
        //     create_cudnn_args_bwd_input<T>(&new_args, kernel, dy, dx, strides, paddings);
        //     get_object_pool()->regist(key, new_args);
        //     args_ptr = get_object_pool()->search<cudnn_conv2d_bwd_input_arguments>(key);
        // }
        cudnn_conv2d_bwd_input_arguments new_args;
        create_cudnn_args_bwd_input<T>(&new_args, kernel, dy, dx, strides, paddings);
        cudnn_conv2d_bwd_input_arguments *args_ptr = &new_args;

        const float alpha = 1, beta = 0;
        ASSERT_SUCCESS(cudnnConvolutionBackwardData(
            args_ptr->_handle,
            &alpha,
            args_ptr->_w_desc, kernel.device_data<void>(),
            args_ptr->_dy_desc, dy.device_data<void>(),
            args_ptr->_conv_desc, args_ptr->_data_algo,
            args_ptr->_ws_data, args_ptr->_ws_data_size,
            &beta,
            args_ptr->_dx_desc, dx.mutable_device_data<void>()
        ));
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
        // std::string key =
        //     "conv2d_bwd_kernel/" + 
        //     std::to_string(x.get_node().get_order()) +
        //     std::to_string(dkernel.size()) +
        //     std::to_string(strides[0]) +
        //     std::to_string(strides[1]) +
        //     std::to_string(paddings[0]) +
        //     std::to_string(paddings[1]);
        // auto args_ptr = get_object_pool()->search<cudnn_conv2d_bwd_kernel_arguments>(key);
        // if(!args_ptr){
        //     cudnn_conv2d_bwd_kernel_arguments new_args;
        //     create_cudnn_args_bwd_kernel<T>(&new_args, x, dy, dkernel, strides, paddings);
        //     get_object_pool()->regist(key, new_args);
        //     args_ptr = get_object_pool()->search<cudnn_conv2d_bwd_kernel_arguments>(key);
        // }
        cudnn_conv2d_bwd_kernel_arguments new_args;
        create_cudnn_args_bwd_kernel<T>(&new_args, x, dy, dkernel, strides, paddings);
        cudnn_conv2d_bwd_kernel_arguments *args_ptr = &new_args;

        const float alpha = 1, beta = 0;

        ASSERT_SUCCESS(cudnnConvolutionBackwardFilter(
            args_ptr->_handle,
            &alpha, args_ptr->_x_desc, x.device_data<void>(),
            args_ptr->_dy_desc, dy.device_data<void>(),
            args_ptr->_conv_desc, args_ptr->_filter_algo,
            args_ptr->_ws_filter, args_ptr->_ws_filter_size,
            &beta,
            args_ptr->_dw_desc, dkernel.mutable_device_data<void>()
        ));
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
