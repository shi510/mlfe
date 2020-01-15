#include "mlfe/core/op_algo.h"
#include "mlfe/math/activations.h"
#include "mlfe/device_context/cuda_context.h"

namespace mlfe{
namespace algorithm_cuda{

template <class Tp>
class ReLU : public OpAlgo{
using T = typename Tp::T;
public:
    ReLU(OpAlgoContext *oac) : OpAlgo(oac, "ReLU"){
        y = oac->get_output(0);
        x = y.get_children()[0];
        size = x.size();
    }

    void Compute() override{
        auto x_ptr = x.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        math::relu<T, CUDAContext>(
            size,
            x_ptr,
            y_ptr
            );
    }
private:
    Tensor x;
    Tensor y;
    int size;
};

REGIST_OP_ALGO(ReLU)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReLU<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class ReLUGrad : public OpAlgo{
using T = typename Tp::T;
public:
    ReLUGrad(OpAlgoContext *oac) : OpAlgo(oac){
        dx = oac->get_output(0);
        x = dx.get_children()[0];
        dy = dx.get_children()[2];
        size = x.size();
    }

    void Compute() override{
        auto x_ptr = x.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();

        math::relu_gradient<T, CUDAContext>(
            size,
            x_ptr,
            dy_ptr,
            dx_ptr
            );
    }

private:
    Tensor x;
    Tensor dy;
    Tensor dx;
    int size;
};

REGIST_OP_GRAD_ALGO(ReLU)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReLUGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class Sigmoid : public OpAlgo{
using T = typename Tp::T;
public:
    Sigmoid(OpAlgoContext *oac) : OpAlgo(oac, "Sigmoid"){
        y = oac->get_output(0);
        x = y.get_children()[0];
        size = x.size();
    }

    void Compute() override{
        auto x_ptr = x.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        math::sigmoid<T, CUDAContext>(
            size,
            x_ptr,
            y_ptr
            );
    }
private:
    Tensor x;
    Tensor y;
    int size;
};

REGIST_OP_ALGO(Sigmoid)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Sigmoid<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class SigmoidGrad : public OpAlgo{
using T = typename Tp::T;
public:
    SigmoidGrad(OpAlgoContext *oac) : OpAlgo(oac){
        dx = oac->get_output(0);
        y = dx.get_children()[1];
        dy = dx.get_children()[2];
        size = y.size();
    }

    void Compute() override{
        auto y_ptr = y.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();

        math::sigmoid_gradient<T, CUDAContext>(
            size,
            y_ptr,
            dy_ptr,
            dx_ptr
            );
    }

private:
    Tensor y;
    Tensor dy;
    Tensor dx;
    int size;
};

REGIST_OP_GRAD_ALGO(Sigmoid)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SigmoidGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm cuda
} // end namespace mlfe
