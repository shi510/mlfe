#include "../core/op_algo.h"
#include "../core/tensor_mem_ref.h"
#include "../math/activations.h"
#include "../device_context/cuda_context.h"

namespace mlfe{ namespace algorithm_cuda{

template <class Dev, class Tp>
class ReLU : public OpAlgo{
using T = typename Tp::T;
public:
    ReLU(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_input(0);
        y = oac->get_output(0);
        size = x->Size();
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto y_ptr = y->Data<T>();
        math::relu<T, CUDAContext>(
            size,
            x_ptr,
            y_ptr
            );
    }
private:
    TensorMemRef *x;
    TensorMemRef *y;
    int size;
};

REGIST_OP_ALGO(ReLU)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string)
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
        x = oac->get_input(0);
        dy = oac->get_input(2);
        dx = oac->get_output(0);
        size = x->Size();
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto dy_ptr = dy->Data<T>();
        auto dx_ptr = dx->Data<T>();

        math::relu_gradient<T, CUDAContext>(
            size,
            x_ptr,
            dy_ptr,
            dx_ptr
            );
    }

private:
    TensorMemRef *x;
    TensorMemRef *dy;
    TensorMemRef *dx;
    int size;
};

REGIST_OP_GRAD_ALGO(ReLU)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CUDA::string)
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
        x = oac->get_input(0);
        y = oac->get_output(0);
        size = x->Size();
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto y_ptr = y->Data<T>();
        math::sigmoid<T, CUDAContext>(
            size,
            x_ptr,
            y_ptr
            );
    }
private:
    TensorMemRef *x;
    TensorMemRef *y;
    int size;
};

REGIST_OP_ALGO(Sigmoid)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string)
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
        x = oac->get_input(0);
        dy = oac->get_input(2);
        dx = oac->get_output(0);
        size = x->Size();
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto dy_ptr = dy->Data<T>();
        auto dx_ptr = dx->Data<T>();

        math::sigmoid_gradient<T, CUDAContext>(
            size,
            x_ptr,
            dy_ptr,
            dx_ptr
            );
    }

private:
    TensorMemRef *x;
    TensorMemRef *dy;
    TensorMemRef *dx;
    int size;
};

REGIST_OP_GRAD_ALGO(Sigmoid)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SigmoidGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm cuda
} // end namespace mlfe
