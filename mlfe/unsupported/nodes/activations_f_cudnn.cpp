#include <cudnn.h>
#include "../core/node.hpp"
#include "../../device_context/cuda_context.hpp"
#include "../../utils/assert.hpp"
#include "../../math/functions.hpp"
#include "activations.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CUDAContext>
struct ActivationCudnnF : NodeFunctor {
    void inline ASSERT_SUCCESS(cudnnStatus_t t) {
        if (t != CUDNN_STATUS_SUCCESS) {
            std::cout << "NOT CUDNN_STATUS_SUCCESS" << " : " << cudnnGetErrorString(t) << std::endl;
            exit(0);
        }
    }
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        cudnnActivationMode_t act_mode;
        _x = oc->inputs[0];
        _y = oc->outputs[0];
        _type = oc->attr->GetParam<ActivationType>("Type");
        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        _y->Allocate(Accelerator::CUDA, dt);

        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_x_desc);
        cudnnCreateTensorDescriptor(&_y_desc);
        cudnnCreateActivationDescriptor(&_act_desc);
        
        ASSERT_SUCCESS(cudnnSetTensor4dDescriptor(
            _x_desc,
            CUDNN_TENSOR_NCHW,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            1, _x->Size(), 1, 1
        ));

        ASSERT_SUCCESS(cudnnSetTensor4dDescriptor(
            _y_desc,
            CUDNN_TENSOR_NCHW,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            1, _y->Size(), 1, 1        ));        switch (_type) {
        case ActivationType::ReLU:
            act_mode = CUDNN_ACTIVATION_RELU;
            break;
        case ActivationType::Sigmoid:
            act_mode = CUDNN_ACTIVATION_SIGMOID;
            break;
        }        ASSERT_SUCCESS(cudnnSetActivationDescriptor(            _act_desc,            act_mode,            CUDNN_PROPAGATE_NAN,            0.        ));
    }

    void Run() override {
        const float alpha = 1, beta = 0;
        ASSERT_SUCCESS(
            cudnnActivationForward(
                _handle,
                _act_desc,
                &alpha,
                _x_desc, _x->GetPtr<T>(),
                &beta,
                _y_desc, _y->GetPtr<T>()
            )
        );
    }

    ~ActivationCudnnF() {
        cudnnDestroyTensorDescriptor(_x_desc);
        cudnnDestroyTensorDescriptor(_y_desc);
        cudnnDestroyActivationDescriptor(_act_desc);
        cudnnDestroy(_handle);
    }

    Tensor *_x;
    Tensor *_y;
    cudnnHandle_t _handle;
    cudnnTensorDescriptor_t _x_desc;
    cudnnTensorDescriptor_t _y_desc;
    cudnnActivationDescriptor_t _act_desc;
    ActivationType _type;
};

REGIST_NODE_FUNCTOR(Activation, DataType::F32, Accelerator::CUDNN, ActivationCudnnF<float>)
//REGIST_NODE_FUNCTOR(Activation, DataType::F64, Accelerator::CUDNN, ActivationCudnnF<double>)

template <typename T, typename D = CUDAContext>
struct ActivationGradCudnnF : NodeFunctor {
    void inline ASSERT_SUCCESS(cudnnStatus_t t) {
        if (t != CUDNN_STATUS_SUCCESS) {
            std::cout << "NOT CUDNN_STATUS_SUCCESS" << " : " << cudnnGetErrorString(t) << std::endl;
            exit(0);
        }
    }
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        cudnnActivationMode_t act_mode;
        _x = oc->inputs[0];
        _y = oc->inputs[1];
        _dy = oc->inputs[2];
        _dx = oc->outputs[0];
        _type = oc->attr->GetParam<ActivationType>("Type");
        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        _dx->Allocate(Accelerator::CUDA, dt);
        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_dx_desc);
        cudnnCreateTensorDescriptor(&_dy_desc);
        cudnnCreateActivationDescriptor(&_act_desc);

        ASSERT_SUCCESS(cudnnSetTensor4dDescriptor(
            _dx_desc,
            CUDNN_TENSOR_NCHW,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            1, _dx->Size(), 1, 1
        ));

        ASSERT_SUCCESS(cudnnSetTensor4dDescriptor(
            _dy_desc,
            CUDNN_TENSOR_NCHW,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            1, _dy->Size(), 1, 1        ));        switch (_type) {
        case ActivationType::ReLU:
            act_mode = CUDNN_ACTIVATION_RELU;
            break;
        case ActivationType::Sigmoid:
            act_mode = CUDNN_ACTIVATION_SIGMOID;
            break;
        }        ASSERT_SUCCESS(cudnnSetActivationDescriptor(            _act_desc,            act_mode,            CUDNN_PROPAGATE_NAN,            0.        ));
    }

    void Run() override {
        const float alpha = 1, beta = 0;
        ASSERT_SUCCESS(
            cudnnActivationBackward(
                _handle,
                _act_desc,
                &alpha,
                _dy_desc, _y->GetPtr<T>(),
                _dy_desc, _dy->GetPtr<T>(),
                _dx_desc, _x->GetPtr<T>(),
                &beta,
                _dx_desc, _dx->GetPtr<T>()
            )
        );
    }

    ~ActivationGradCudnnF() {
        cudnnDestroyTensorDescriptor(_dx_desc);
        cudnnDestroyTensorDescriptor(_dy_desc);
        cudnnDestroyActivationDescriptor(_act_desc);
        cudnnDestroy(_handle);
    }

    Tensor *_x, *_y, *_dy;
    Tensor *_dx;
    cudnnHandle_t _handle;
    cudnnTensorDescriptor_t _dx_desc;
    cudnnTensorDescriptor_t _dy_desc;
    cudnnActivationDescriptor_t _act_desc;
    ActivationType _type;
};

REGIST_NODE_GRADIENT_FUNCTOR(Activation, DataType::F32, Accelerator::CUDNN, ActivationGradCudnnF<float>)
//REGIST_NODE_GRADIENT_FUNCTOR(Activation, DataType::F64, Accelerator::CUDNN, ActivationCudnnF<double>)

} // end namespace node
} // end namespace mlfe