#include "mlfe/operators/batch_norm2d.h"
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
        std::string e_msg = "NOT CUDNN_STATUS_SUCCESS : ";
        e_msg += cudnnGetErrorString(t);
        std::cout<<"Fail CUDNN_STATUS_SUCCESS"<<std::endl;
        throw std::runtime_error(e_msg);
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
struct batch_norm2d_nhwc_op
{
    static void run(
        Tensor x,
        Tensor scales,
        Tensor biases,
        Tensor rmean,
        Tensor rvar,
        Tensor y,
        bool track_running_status)
    {
        cudnnHandle_t _handle = cuda_context_v2::create()->get_cudnn_handle();
        cudnnTensorDescriptor_t _dst_desc;
        cudnnTensorDescriptor_t _norm_desc;
        auto x_shape = x.shape();
        auto saved_mean_ptr = create_memory(x_shape[3] * sizeof(T));
        auto saved_var_ptr = create_memory(x_shape[3] * sizeof(T));
        cudnnCreateTensorDescriptor(&_dst_desc);
        cudnnCreateTensorDescriptor(&_norm_desc);
        cudnnSetTensor4dDescriptor(_dst_desc,
            CUDNN_TENSOR_NHWC,
            get_cudnn_type<T>(),
            x_shape[0], x_shape[3], x_shape[1], x_shape[2]);
        cudnnSetTensor4dDescriptor(_norm_desc,
            CUDNN_TENSOR_NHWC, get_cudnn_type<T>(), 1, x_shape[3], 1, 1);
        x.get_node().add_attr("saved_mean", saved_mean_ptr);
        x.get_node().add_attr("saved_var", saved_var_ptr);
        const T one = 1;
        const T zero = 0;
        if(track_running_status){
            ASSERT_SUCCESS(cudnnBatchNormalizationForwardTraining(_handle,
                CUDNN_BATCHNORM_SPATIAL,
                &one, &zero,
                _dst_desc,
                x.device_data<void>(),
                _dst_desc,
                y.mutable_device_data<void>(),
                _norm_desc,
                scales.device_data<void>(),
                biases.device_data<void>(),
                1e-2,
                rmean.mutable_device_data<void>(),
                rvar.mutable_device_data<void>(),
                1e-5,
                saved_mean_ptr->mutable_device_data<void>(),
                saved_var_ptr->mutable_device_data<void>()));
        }
        else{
            ASSERT_SUCCESS(cudnnBatchNormalizationForwardInference(_handle,
                CUDNN_BATCHNORM_SPATIAL,
                &one,
                &zero,
                _dst_desc,
                x.device_data<void>(),
                _dst_desc,
                y.mutable_device_data<void>(),
                _norm_desc,
                scales.device_data<void>(),
                biases.device_data<void>(),
                rmean.device_data<void>(),
                rvar.device_data<void>(),
                1e-5));
        }

        cudnnDestroyTensorDescriptor(_dst_desc);
        cudnnDestroyTensorDescriptor(_norm_desc);
    }
};

template <class T>
struct batch_norm2d_nhwc_grad_op
{
    static void run(
        Tensor x,
        Tensor scales,
        Tensor dy,
        Tensor dx,
        Tensor dscales,
        Tensor dbiases)
    {
        cudnnHandle_t _handle = cuda_context_v2::create()->get_cudnn_handle();
        cudnnTensorDescriptor_t _dst_desc;
        cudnnTensorDescriptor_t _norm_desc;
        auto x_shape = x.shape();
        auto saved_mean_ptr = *x.get_node().get_attr("saved_mean").data<memory_ptr>();
        auto saved_var_ptr = *x.get_node().get_attr("saved_var").data<memory_ptr>();
        cudnnCreateTensorDescriptor(&_dst_desc);
        cudnnCreateTensorDescriptor(&_norm_desc);
        cudnnSetTensor4dDescriptor(_dst_desc,
            CUDNN_TENSOR_NHWC,
            get_cudnn_type<T>(),
            x_shape[0], x_shape[3], x_shape[1], x_shape[2]);
        cudnnSetTensor4dDescriptor(_norm_desc,
            CUDNN_TENSOR_NHWC, get_cudnn_type<T>(), 1, x_shape[3], 1, 1);
        const T one = 1;
        const T zero = 0;
        ASSERT_SUCCESS(cudnnBatchNormalizationBackward(_handle,
            CUDNN_BATCHNORM_SPATIAL,
            &one,
            &zero,
            &one,
            &one,
            _dst_desc,
            x.device_data<void>(),
            _dst_desc,
            dy.device_data<void>(),
            _dst_desc,
            dx.mutable_device_data<void>(),
            _norm_desc,
            scales.device_data<void>(),
            dscales.mutable_device_data<void>(),
            dbiases.mutable_device_data<void>(),
            1e-5,
            saved_mean_ptr->device_data<void>(),
            saved_var_ptr->device_data<void>()));
        cudnnDestroyTensorDescriptor(_dst_desc);
        cudnnDestroyTensorDescriptor(_norm_desc);
    }
};

} // namespace anonymous

REGIST_OP_KERNEL(
    batch_norm2d_fwd,
    batch_norm_fwd_fn_t,
    batch_norm2d_nhwc_op<float>::run
    );

REGIST_OP_KERNEL(
    batch_norm2d_bwd,
    batch_norm_bwd_fn_t,
    batch_norm2d_nhwc_grad_op<float>::run
    );

} // namespace operators
} // namespace mlfe
