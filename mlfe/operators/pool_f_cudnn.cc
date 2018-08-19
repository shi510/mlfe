#include "../core/op_algo.h"
#include "../core/device.h"
#include "../core/tensor_mem_ref.h"
#include "../math/blas.h"
#include "../math/transform.h"
#include "../device_context/cuda_context.h"
#include <cudnn.h>
#include <string>

namespace mlfe{

template <class Dev, class Tp>
class MaxPool : public OpAlgo{
using T = typename Tp::T;
public:
    void inline ASSERT_SUCCESS(cudnnStatus_t t){
        if(t != CUDNN_STATUS_SUCCESS){
            std::string e_msg;
            e_msg = "NOT CUDNN_STATUS_SUCCESS : ";
            e_msg += cudnnGetErrorString(t);
            throw e_msg;
        }
    }

    MaxPool(OpAlgoContext *oac) : OpAlgo(oac){
        using IntVec = std::vector<type::int32::T>;
        x = oac->GetVar("X");
        y = oac->GetVar("Y");
        filters_hw = oac->GetAttr<IntVec>("filters_hw");
        strides = oac->GetAttr<IntVec>("strides");
        pads = oac->GetAttr<IntVec>("pads");
        auto cudnn_type = [](std::string type_str){
            return type_str == type::float32::string ?
                CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
        };

        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_x_desc);
        cudnnCreateTensorDescriptor(&_y_desc);
        cudnnCreatePoolingDescriptor(&_pooling_desc);

        cudnnSetTensor4dDescriptor(
            _x_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            x->Shape()[0], x->Shape()[1], x->Shape()[2], x->Shape()[3]
        );

        cudnnSetTensor4dDescriptor(
            _y_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            y->Shape()[0], y->Shape()[1], y->Shape()[2], y->Shape()[3]
        );

        ASSERT_SUCCESS(cudnnSetPooling2dDescriptor(
            _pooling_desc,
            CUDNN_POOLING_MAX,
            CUDNN_PROPAGATE_NAN,
            filters_hw[0], filters_hw[1],
            0, 0,
            strides[0], strides[1]
        ));
    }

    void Compute() override{
        const float alpha = 1, beta = 0;
        ASSERT_SUCCESS(cudnnPoolingForward(
            _handle,
            _pooling_desc,
            &alpha,
            _x_desc, x->Data<T>(),
            &beta,
            _y_desc, y->Data<T>()
        ));
    }

    ~MaxPool(){
        cudnnDestroyTensorDescriptor(_x_desc);
        cudnnDestroyTensorDescriptor(_y_desc);
        cudnnDestroyPoolingDescriptor(_pooling_desc);
        cudnnDestroy(_handle);
    }

private:
    TensorMemRef *x;
    TensorMemRef *y;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
    cudnnHandle_t _handle;
    cudnnTensorDescriptor_t _x_desc;
    cudnnTensorDescriptor_t _y_desc;
    cudnnPoolingDescriptor_t _pooling_desc;
};

REGIST_OP_ALGO(MaxPool)
    .Input("X", type::float32::string)
    .Output("IDX", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string_cudnn)
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = MaxPool<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class MaxPoolGrad : public OpAlgo{
using T = typename Tp::T;
public:
    void inline ASSERT_SUCCESS(cudnnStatus_t t){
        if(t != CUDNN_STATUS_SUCCESS){
            std::string e_msg;
            e_msg = "NOT CUDNN_STATUS_SUCCESS : ";
            e_msg += cudnnGetErrorString(t);
            throw e_msg;
        }
    }

    MaxPoolGrad(OpAlgoContext *oac) : OpAlgo(oac){
        using IntVec = std::vector<type::int32::T>;
        x = oac->GetVar("X");
        y = oac->GetVar("Y");
        dy = oac->GetVar("dY");
        dx = oac->GetVar("dX");
        filters_hw = oac->GetAttr<IntVec>("filters_hw");
        strides = oac->GetAttr<IntVec>("strides");
        pads = oac->GetAttr<IntVec>("pads");
        auto cudnn_type = [](std::string type_str){
            return type_str == type::float32::string ?
                CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
        };

        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_dx_desc);
        cudnnCreateTensorDescriptor(&_dy_desc);
        cudnnCreatePoolingDescriptor(&_pooling_desc);

        cudnnSetTensor4dDescriptor(
            _dx_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            x->Shape()[0], x->Shape()[1], x->Shape()[2], x->Shape()[3]
        );

        cudnnSetTensor4dDescriptor(
            _dy_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            dy->Shape()[0], dy->Shape()[1], dy->Shape()[2], dy->Shape()[3]
        );

        ASSERT_SUCCESS(cudnnSetPooling2dDescriptor(
            _pooling_desc,
            CUDNN_POOLING_MAX,
            CUDNN_PROPAGATE_NAN,
            filters_hw[0], filters_hw[1],
            0, 0,
            strides[0], strides[1]
        ));
    }

    void Compute() override{
        const float alpha = 1, beta = 0;
        ASSERT_SUCCESS(cudnnPoolingBackward(
            _handle,
            _pooling_desc,
            &alpha,
            _dy_desc, y->Data<T>(),
            _dy_desc, dy->Data<T>(),
            _dx_desc, x->Data<T>(),
            &beta,
            _dx_desc, dx->Data<T>()
        ));
    }

    ~MaxPoolGrad(){
        cudnnDestroyTensorDescriptor(_dx_desc);
        cudnnDestroyTensorDescriptor(_dy_desc);
        cudnnDestroyPoolingDescriptor(_pooling_desc);
        cudnnDestroy(_handle);
    }

private:
    TensorMemRef *x;
    TensorMemRef *y;
    TensorMemRef *dy;
    TensorMemRef *dx;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
    cudnnHandle_t _handle;
    cudnnTensorDescriptor_t _dx_desc;
    cudnnTensorDescriptor_t _dy_desc;
    cudnnPoolingDescriptor_t _pooling_desc;
};

REGIST_OP_GRAD_ALGO(MaxPool)
    .Input("X", type::float32::string)
    .Input("IDX", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CUDA::string_cudnn)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = MaxPoolGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace mlfe
