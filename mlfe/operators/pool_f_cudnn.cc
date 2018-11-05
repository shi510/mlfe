#include "../core/op_algo.h"
#include "../core/device.h"
#include "../math/blas.h"
#include "../math/transform.h"
#include "../device_context/cuda_context.h"
#include <cudnn.h>
#include <string>

namespace mlfe{
namespace algorithm_cudnn{

template <class Tp, cudnnPoolingMode_t PoolingMode>
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

    MaxPool(OpAlgoContext *oac) : OpAlgo(oac, "MaxPool"){
        using IntVec = std::vector<type::int32::T>;
        y = oac->get_output(0);
        x = y.get_children()[0];
        filters_hw = oac->get_attr<IntVec>("kernel");
        strides = oac->get_attr<IntVec>("stride");
        pads = oac->get_attr<IntVec>("padding");
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
            x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
        );

        cudnnSetTensor4dDescriptor(
            _y_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            y.Shape()[0], y.Shape()[1], y.Shape()[2], y.Shape()[3]
        );

        ASSERT_SUCCESS(cudnnSetPooling2dDescriptor(
            _pooling_desc,
            PoolingMode,
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
            _x_desc, x.device_data<void>(),
            &beta,
            _y_desc, y.mutable_device_data<void>()
        ));
    }

    ~MaxPool(){
        cudnnDestroyTensorDescriptor(_x_desc);
        cudnnDestroyTensorDescriptor(_y_desc);
        cudnnDestroyPoolingDescriptor(_pooling_desc);
        cudnnDestroy(_handle);
    }

private:
    Tensor x;
    Tensor y;
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
    .Device("CUDA(CUDNN)")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = MaxPool<type::float32, CUDNN_POOLING_MAX>;
        return std::make_shared<T>(oac);
    })
    .Finish();

REGIST_OP_ALGO(AvgPool)
    .Input("X", type::float32::string)
    .Output("IDX", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CUDA(CUDNN)")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = MaxPool<type::float32, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp, cudnnPoolingMode_t PoolingMode>
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

    MaxPoolGrad(OpAlgoContext *oac) : OpAlgo(oac, "MaxPoolGradient"){
        using IntVec = std::vector<type::int32::T>;
        dx = oac->get_output(0);
        x = dx.get_children()[0];
        y = dx.get_children()[2];
        dy = dx.get_children()[3];
        filters_hw = oac->get_attr<IntVec>("kernel");
        strides = oac->get_attr<IntVec>("stride");
        pads = oac->get_attr<IntVec>("padding");
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
            dx.Shape()[0], dx.Shape()[1], dx.Shape()[2], dx.Shape()[3]
        );

        cudnnSetTensor4dDescriptor(
            _dy_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            dy.Shape()[0], dy.Shape()[1], dy.Shape()[2], dy.Shape()[3]
        );

        ASSERT_SUCCESS(cudnnSetPooling2dDescriptor(
            _pooling_desc,
            PoolingMode,
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
            _dy_desc, y.device_data<void>(),
            _dy_desc, dy.device_data<void>(),
            _dx_desc, x.device_data<void>(),
            &beta,
            _dx_desc, dx.mutable_device_data<void>()
        ));
    }

    ~MaxPoolGrad(){
        cudnnDestroyTensorDescriptor(_dx_desc);
        cudnnDestroyTensorDescriptor(_dy_desc);
        cudnnDestroyPoolingDescriptor(_pooling_desc);
        cudnnDestroy(_handle);
    }

private:
    Tensor x;
    Tensor y;
    Tensor dy;
    Tensor dx;
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
    .Device("CUDA(CUDNN)")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = MaxPoolGrad<type::float32, CUDNN_POOLING_MAX>;
        return std::make_shared<T>(oac);
    })
    .Finish();

REGIST_OP_GRAD_ALGO(AvgPool)
    .Input("X", type::float32::string)
    .Input("IDX", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CUDA(CUDNN)")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = MaxPoolGrad<type::float32, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cudnn
} // end namespace mlfe
