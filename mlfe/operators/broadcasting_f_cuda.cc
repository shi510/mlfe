#include "mlfe/core/op_algo.h"
#include "mlfe/device_context/cuda_context.h"
#include "mlfe/math/transform.h"
#include "mlfe/math/basic_functions.h"
#include <iostream>

namespace mlfe{
namespace algorithm_cuda{

template <class Tp>
class Broadcasting : public OpAlgo{
    using T = typename Tp::T;
public:
    Broadcasting(OpAlgoContext *oac) : OpAlgo(oac, "Broadcasting"){
        y = oac->get_output(0);
        x = oac->get_input(0);
        if (x.dims() > 4 || y.dims() > 4) {
            throw std::runtime_error("broadcasting only supports upto 4 dimensions.");
        }
        x_shape.resize(4);
        y_shape.resize(4);
        std::fill(x_shape.begin(), x_shape.end(), 1);
        std::fill(y_shape.begin(), y_shape.end(), 1);
        std::copy(x.shape().begin(), x.shape().end(), x_shape.begin());
        std::copy(y.shape().begin(), y.shape().end(), y_shape.begin());
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = x.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();

        math::broadcast<T, CUDAContext>(x_ptr, y_ptr,
            x_shape[0], x_shape[1], x_shape[2], x_shape[3],
            y_shape[0], y_shape[1], y_shape[2], y_shape[3]);
    }
private:
    Tensor x;
    Tensor y;
    std::vector<int> x_shape;
    std::vector<int> y_shape;
};

REGIST_OP_ALGO(Broadcasting)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Broadcasting<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class BroadcastingGrad : public OpAlgo{
    using T = typename Tp::T;
public:
    BroadcastingGrad(OpAlgoContext *oac) : OpAlgo(oac){
        dx = oac->get_output(0);
        dy = oac->get_input(0);
        if (dy.dims() > 4 || dx.dims() > 4) {
            throw std::runtime_error("broadcasting only supports upto 4 dimensions.");
        }
        dy_shape.resize(4);
        dx_shape.resize(4);
        std::fill(dy_shape.begin(), dy_shape.end(), 1);
        std::fill(dx_shape.begin(), dx_shape.end(), 1);
        std::copy(dy.shape().begin(), dy.shape().end(), dy_shape.begin());
        std::copy(dx.shape().begin(), dx.shape().end(), dx_shape.begin());
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();

        // zero
        math::set<T, CUDAContext>(dx.size(), T(0), dx_ptr);

        math::broadcast_gradient<T, CUDAContext>(dy_ptr, dx_ptr,
            dy_shape[0], dy_shape[1], dy_shape[2], dy_shape[3],
            dx_shape[0], dx_shape[1], dx_shape[2], dx_shape[3]);
    }

private:
    Tensor dy;
    Tensor dx;
    std::vector<int> dy_shape;
    std::vector<int> dx_shape;
};

REGIST_OP_GRAD_ALGO(Broadcasting)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = BroadcastingGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
