#include <cudnn.h>
#include "../core/node.hpp"
#include "../../device_context/cuda_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions.hpp"
#include "../../utils/assert.hpp"
#include "../../math/functions_cuda.hpp"
#include "../../math/transform.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CUDAContext>
struct MaxPoolCudnnF : NodeFunctor {
    void inline ASSERT_SUCCESS(cudnnStatus_t t) {
        if (t != CUDNN_STATUS_SUCCESS) {
            std::cout << "NOT CUDNN_STATUS_SUCCESS" << " : " << cudnnGetErrorString(t) << std::endl;
            exit(0);
        }
    }
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        _kernel = oc->attr->GetParam<std::vector<int>>("Kernel");
        _stride = oc->attr->GetParam<std::vector<int>>("Stride");
        _x = oc->inputs[0];
        _y = oc->outputs[0];

        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }

        _y->Allocate(Accelerator::CUDA, dt);
        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_x_desc);
        cudnnCreateTensorDescriptor(&_y_desc);
        cudnnCreatePoolingDescriptor(&_pooling_desc);

        cudnnSetTensor4dDescriptor(
            _x_desc,
            CUDNN_TENSOR_NCHW,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            _x->Dim(0), _x->Dim(1), _x->Dim(2), _x->Dim(3)
        );

        cudnnSetTensor4dDescriptor(
            _y_desc,
            CUDNN_TENSOR_NCHW,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            _y->Dim(0), _y->Dim(1), _y->Dim(2), _y->Dim(3)
        );

        ASSERT_SUCCESS(cudnnSetPooling2dDescriptor(
            _pooling_desc,
            CUDNN_POOLING_MAX,
            CUDNN_PROPAGATE_NAN,
            _kernel[0], _kernel[1],
            0, 0,
            _stride[0], _stride[1]
        ));
    }

    void Run() override {
        const float alpha = 1, beta = 0;
        ASSERT_SUCCESS(cudnnPoolingForward(
            _handle,
            _pooling_desc,
            &alpha,
            _x_desc, _x->GetPtr<T>(),
            &beta,
            _y_desc, _y->GetPtr<T>()
        ));
    }

    ~MaxPoolCudnnF() {
        cudnnDestroyTensorDescriptor(_x_desc);
        cudnnDestroyTensorDescriptor(_y_desc);
        cudnnDestroyPoolingDescriptor(_pooling_desc);
        cudnnDestroy(_handle);
    }

    Tensor *_x;
    Tensor *_y;
    std::vector<int> _kernel;
    std::vector<int> _stride;
    cudnnHandle_t _handle;
    cudnnTensorDescriptor_t _x_desc;
    cudnnTensorDescriptor_t _y_desc;
    cudnnPoolingDescriptor_t _pooling_desc;
};

REGIST_NODE_FUNCTOR(MaxPool, DataType::F32, Accelerator::CUDNN, MaxPoolCudnnF<float>)
REGIST_NODE_FUNCTOR(MaxPool, DataType::F64, Accelerator::CUDNN, MaxPoolCudnnF<double>)

template <typename T, typename D = CUDAContext>
struct MaxPoolGradCudnnF : NodeFunctor {
    void inline ASSERT_SUCCESS(cudnnStatus_t t) {
        if (t != CUDNN_STATUS_SUCCESS) {
            std::cout << "NOT CUDNN_STATUS_SUCCESS" << " : " << cudnnGetErrorString(t) << std::endl;
            exit(0);
        }
    }
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        _kernel = oc->attr->GetParam<std::vector<int>>("Kernel");
        _stride = oc->attr->GetParam<std::vector<int>>("Stride");
        _x = oc->inputs[0];
        _y = oc->inputs[1];
        _dy = oc->inputs[3];
        _dx = oc->outputs[0];
        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        _dx->Allocate(Accelerator::CUDA, dt);

        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_dx_desc);
        cudnnCreateTensorDescriptor(&_dy_desc);
        cudnnCreatePoolingDescriptor(&_pooling_desc);

        cudnnSetTensor4dDescriptor(
            _dx_desc,
            CUDNN_TENSOR_NCHW,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            _x->Dim(0), _x->Dim(1), _x->Dim(2), _x->Dim(3)
        );

        cudnnSetTensor4dDescriptor(
            _dy_desc,
            CUDNN_TENSOR_NCHW,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            _y->Dim(0), _y->Dim(1), _y->Dim(2), _y->Dim(3)
        );

        ASSERT_SUCCESS(cudnnSetPooling2dDescriptor(
            _pooling_desc,
            CUDNN_POOLING_MAX,
            CUDNN_PROPAGATE_NAN,
            _kernel[0], _kernel[1],
            0, 0,
            _stride[0], _stride[1]
        ));
    }

    void Run() override {
        const float alpha = 1, beta = 0;
        ASSERT_SUCCESS(cudnnPoolingBackward(
            _handle,
            _pooling_desc,
            &alpha,
            _dy_desc, _y->GetPtr<T>(),
            _dy_desc, _dy->GetPtr<T>(),
            _dx_desc, _x->GetPtr<T>(),
            &beta,
            _dx_desc, _dx->GetPtr<T>()
        ));
    }

    ~MaxPoolGradCudnnF() {
        cudnnDestroyTensorDescriptor(_dx_desc);
        cudnnDestroyTensorDescriptor(_dy_desc);
        cudnnDestroyPoolingDescriptor(_pooling_desc);
        cudnnDestroy(_handle);
    }

    Tensor *_x, *_y, *_dy;
    Tensor *_dx;
    std::vector<int> _kernel;
    std::vector<int> _stride;
    cudnnHandle_t _handle;
    cudnnTensorDescriptor_t _dx_desc;
    cudnnTensorDescriptor_t _dy_desc;
    cudnnPoolingDescriptor_t _pooling_desc;
};

REGIST_NODE_GRADIENT_FUNCTOR(MaxPool, DataType::F32, Accelerator::CUDNN, MaxPoolGradCudnnF<float>)
REGIST_NODE_GRADIENT_FUNCTOR(MaxPool, DataType::F64, Accelerator::CUDNN, MaxPoolGradCudnnF<double>)

} // end namespace node
} // end namespace mlfe
