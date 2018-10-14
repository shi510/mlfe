#include "../core/op_algo.h"
#include "../core/tensor_mem_ref.h"
#include "../math/activations.h"
#include <cudnn.h>

namespace mlfe{ namespace algorithm_cudnn{

template <class Dev, class Tp>
class ReLU : public OpAlgo{
using T = typename Tp::T;
public:
    ReLU(OpAlgoContext *oac) : OpAlgo(oac){
        auto cudnn_type = [](std::string type_str){
            return type_str == type::float32::string ?
                CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
        };
        std::vector<int> shape(4);
        x = oac->get_input(0);
        y = oac->get_output(0);
        std::fill(shape.begin(), shape.end(), 1);
        for(int n = 0; n < x->Shape().size(); ++n){
            shape[n] = x->Shape()[n];
        }
        cudnnCreate(&handle);
        cudnnCreateTensorDescriptor(&x_desc);
        cudnnCreateActivationDescriptor(&act_desc);

        cudnnSetTensor4dDescriptor(
            x_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            shape[0], shape[1], shape[2], shape[3]);

        cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 1e+4);
    }

    void Compute() override{
        const T alpha = T(1);
        const T beta = T(0);
        auto x_ptr = x->Data<T>();
        auto y_ptr = y->Data<T>();
        cudnnActivationForward(handle, act_desc, &alpha, x_desc, x_ptr, &beta, x_desc, y_ptr);
    }

    ~ReLU(){
        cudnnDestroy(handle);
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyActivationDescriptor(act_desc);
    }

private:
    TensorMemRef *x;
    TensorMemRef *y;
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t x_desc;
    cudnnActivationDescriptor_t act_desc;
};

REGIST_OP_ALGO(ReLU)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string_cudnn)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReLU<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class ReLUGrad : public OpAlgo{
using T = typename Tp::T;
public:
    ReLUGrad(OpAlgoContext *oac) : OpAlgo(oac){
        auto cudnn_type = [](std::string type_str){
            return type_str == type::float32::string ?
                CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
        };
        std::vector<int> shape(4);
        x = oac->get_input(0);
        y = oac->get_input(1);
        dy = oac->get_input(2);
        dx = oac->get_output(0);
        std::fill(shape.begin(), shape.end(), 1);
        for(int n = 0; n < x->Shape().size(); ++n){
            shape[n] = x->Shape()[n];
        }
        cudnnCreate(&handle);
        cudnnCreateTensorDescriptor(&x_desc);
        cudnnCreateActivationDescriptor(&act_desc);

        cudnnSetTensor4dDescriptor(
            x_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            shape[0], shape[1], shape[2], shape[3]);

        cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 1e+4);
    }

    void Compute() override{
        const T alpha = T(1);
        const T beta = T(0);
        auto x_ptr = x->Data<T>();
        auto y_ptr = y->Data<T>();
        auto dy_ptr = dy->Data<T>();
        auto dx_ptr = dx->Data<T>();

        cudnnActivationBackward(
            handle,
            act_desc, &alpha,
            x_desc, y_ptr,
            x_desc, dy_ptr,
            x_desc, x_ptr, &beta,
            x_desc, dx_ptr);
    }

    ~ReLUGrad(){
        cudnnDestroy(handle);
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyActivationDescriptor(act_desc);
    }

private:
    TensorMemRef *x;
    TensorMemRef *y;
    TensorMemRef *dy;
    TensorMemRef *dx;
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t x_desc;
    cudnnActivationDescriptor_t act_desc;
};

REGIST_OP_GRAD_ALGO(ReLU)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CUDA::string_cudnn)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReLUGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class Sigmoid : public OpAlgo{
using T = typename Tp::T;
public:
    Sigmoid(OpAlgoContext *oac) : OpAlgo(oac){
        auto cudnn_type = [](std::string type_str){
            return type_str == type::float32::string ?
                CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
        };
        std::vector<int> shape(4);
        x = oac->get_input(0);
        y = oac->get_output(0);
        std::fill(shape.begin(), shape.end(), 1);
        for(int n = 0; n < x->Shape().size(); ++n){
            shape[n] = x->Shape()[n];
        }

        cudnnCreate(&handle);
        cudnnCreateTensorDescriptor(&x_desc);
        cudnnCreateTensorDescriptor(&y_desc);
        cudnnCreateActivationDescriptor(&act_desc);

        cudnnSetTensor4dDescriptor(
            x_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            shape[0], shape[1], shape[2], shape[3]);

        cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0);
    }

    void Compute() override{
        const T alpha = T(1);
        const T beta = T(0);
        auto x_ptr = x->Data<T>();
        auto y_ptr = y->Data<T>();
        cudnnActivationForward(handle, act_desc, &alpha, x_desc, x_ptr, &beta, x_desc, y_ptr);
    }

    ~Sigmoid(){
        cudnnDestroy(handle);
        cudnnDestroyActivationDescriptor(act_desc);
        cudnnDestroyTensorDescriptor(x_desc);
    }

private:
    TensorMemRef *x;
    TensorMemRef *y;
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t x_desc;
    cudnnTensorDescriptor_t y_desc;
    cudnnActivationDescriptor_t act_desc;
};

REGIST_OP_ALGO(Sigmoid)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string_cudnn)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Sigmoid<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class SigmoidGrad : public OpAlgo{
using T = typename Tp::T;
public:
    SigmoidGrad(OpAlgoContext *oac) : OpAlgo(oac){
        auto cudnn_type = [](std::string type_str){
            return type_str == type::float32::string ?
                CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
        };
        std::vector<int> shape(4);
        x = oac->get_input(0);
        y = oac->get_input(1);
        dy = oac->get_input(2);
        dx = oac->get_output(0);
        std::fill(shape.begin(), shape.end(), 1);
        for(int n = 0; n < x->Shape().size(); ++n){
            shape[n] = x->Shape()[n];
        }

        cudnnCreate(&handle);
        cudnnCreateTensorDescriptor(&x_desc);
        cudnnCreateActivationDescriptor(&act_desc);

        cudnnSetTensor4dDescriptor(
            x_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            shape[0], shape[1], shape[2], shape[3]);

        cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0);
    }

    void Compute() override{
        const T alpha = T(1);
        const T beta = T(0);
        auto x_ptr = x->Data<T>();
        auto y_ptr = y->Data<T>();
        auto dy_ptr = dy->Data<T>();
        auto dx_ptr = dx->Data<T>();

        cudnnActivationBackward(
            handle,
            act_desc, &alpha,
            x_desc, y_ptr,
            x_desc, dy_ptr,
            x_desc, x_ptr, &beta,
            x_desc, dx_ptr);
    }

    ~SigmoidGrad(){
        cudnnDestroy(handle);
        cudnnDestroyActivationDescriptor(act_desc);
        cudnnDestroyTensorDescriptor(x_desc);
    }

private:
    TensorMemRef *x;
    TensorMemRef *y;
    TensorMemRef *dy;
    TensorMemRef *dx;
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t x_desc;
    cudnnActivationDescriptor_t act_desc;
};

REGIST_OP_GRAD_ALGO(Sigmoid)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CUDA::string_cudnn)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SigmoidGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cudnn
} // end namespace mlfe
