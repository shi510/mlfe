#include "mlfe/operators/maxpool2d.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/device_context/cuda_context.h"
#include <cfloat>
#include <cudnn.h>
#include <iostream>

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
        cudnnHandle_t _handle = cuda_context_v2::create()->get_cudnn_handle();
        cudnnTensorDescriptor_t _x_desc;
        cudnnTensorDescriptor_t _y_desc;
        cudnnPoolingDescriptor_t _pooling_desc;
        cudnnCreateTensorDescriptor(&_x_desc);
        cudnnCreateTensorDescriptor(&_y_desc);
        cudnnCreatePoolingDescriptor(&_pooling_desc);
        cudnnSetTensor4dDescriptor(
            _x_desc,
            CUDNN_TENSOR_NHWC,
            get_cudnn_type<T>(),
            x.shape()[0], x.shape()[3], x.shape()[1], x.shape()[2]
        );

        cudnnSetTensor4dDescriptor(
            _y_desc,
            CUDNN_TENSOR_NHWC,
            get_cudnn_type<T>(),
            y.shape()[0], y.shape()[3], y.shape()[1], y.shape()[2]
        );

        ASSERT_SUCCESS(cudnnSetPooling2dDescriptor(
            _pooling_desc,
            CUDNN_POOLING_MAX,
            CUDNN_PROPAGATE_NAN,
            psize[0], psize[1],
            0, 0,
            strides[0], strides[1]
        ));

        const float alpha = 1, beta = 0;
        ASSERT_SUCCESS(cudnnPoolingForward(
            _handle,
            _pooling_desc,
            &alpha,
            _x_desc, x.device_data<void>(),
            &beta,
            _y_desc, y.mutable_device_data<void>()
        ));
        cudnnDestroyTensorDescriptor(_x_desc);
        cudnnDestroyTensorDescriptor(_y_desc);
        cudnnDestroyPoolingDescriptor(_pooling_desc);
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
        cudnnHandle_t _handle = cuda_context_v2::create()->get_cudnn_handle();
        cudnnTensorDescriptor_t _dx_desc;
        cudnnTensorDescriptor_t _dy_desc;
        cudnnPoolingDescriptor_t _pooling_desc;
        cudnnCreateTensorDescriptor(&_dx_desc);
        cudnnCreateTensorDescriptor(&_dy_desc);
        cudnnCreatePoolingDescriptor(&_pooling_desc);

        cudnnSetTensor4dDescriptor(
            _dx_desc,
            CUDNN_TENSOR_NHWC,
            get_cudnn_type<T>(),
            dx.shape()[0], dx.shape()[3], dx.shape()[1], dx.shape()[2]
        );

        cudnnSetTensor4dDescriptor(
            _dy_desc,
            CUDNN_TENSOR_NHWC,
            get_cudnn_type<T>(),
            dy.shape()[0], dy.shape()[3], dy.shape()[1], dy.shape()[2]
        );

        ASSERT_SUCCESS(cudnnSetPooling2dDescriptor(
            _pooling_desc,
            CUDNN_POOLING_MAX,
            CUDNN_PROPAGATE_NAN,
            psize[0], psize[1],
            0, 0,
            strides[0], strides[1]
        ));
        const float alpha = 1, beta = 0;
        ASSERT_SUCCESS(cudnnPoolingBackward(
            _handle,
            _pooling_desc,
            &alpha,
            _dy_desc, y.device_data<void>(),
            _dy_desc, dy.device_data<void>(),
            _dx_desc, x.device_data<void>(),
            &beta,
            _dx_desc, dx.mutable_device_data<void>()
        ));
        cudnnDestroyTensorDescriptor(_dx_desc);
        cudnnDestroyTensorDescriptor(_dy_desc);
        cudnnDestroyPoolingDescriptor(_pooling_desc);
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
