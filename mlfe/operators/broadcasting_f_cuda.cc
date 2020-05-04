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
        x = y.get_children()[0];
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = x.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        int Nx, Cx, Hx, Wx;
        int Ny, Cy, Hy, Wy;

        switch(y.dims()){
        case 1:
            Nx = x.dim(0);
            Cx = 1;
            Hx = 1;
            Wx = 1;
            Ny = y.dim(0);
            Cy = 1;
            Hy = 1;
            Wy = 1;
            break;
        case 2:
            Nx = x.dim(0);
            Cx = x.dim(1);
            Hx = 1;
            Wx = 1;
            Ny = y.dim(0);
            Cy = y.dim(1);
            Hy = 1;
            Wy = 1;
            break;
        case 3:
            Nx = x.dim(0);
            Cx = x.dim(1);
            Hx = x.dim(2);
            Wx = 1;
            Ny = y.dim(0);
            Cy = y.dim(1);
            Hy = y.dim(2);
            Wy = 1;
            break;
        case 4:
            Nx = x.dim(0);
            Cx = x.dim(1);
            Hx = x.dim(2);
            Wx = x.dim(3);
            Ny = y.dim(0);
            Cy = y.dim(1);
            Hy = y.dim(2);
            Wy = y.dim(3);
            break;
        default:
            throw std::string("broadcasting only supports upto 4 dimensions.");
        }

        math::broadcast<T, CUDAContext>(x_ptr, y_ptr,
                                        Nx, Cx, Hx, Wx,
                                        Ny, Cy, Hy, Wy);
    }
private:
    Tensor x;
    Tensor y;
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
        dy = dx.get_children()[0];
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();
        int Nx, Cx, Hx, Wx;
        int Ny, Cy, Hy, Wy;

        switch(dy.dims()){
        case 1:
            Nx = dx.dim(0);
            Cx = 1;
            Hx = 1;
            Wx = 1;
            Ny = dy.dim(0);
            Cy = 1;
            Hy = 1;
            Wy = 1;
            break;
        case 2:
            Nx = dx.dim(0);
            Cx = dx.dim(1);
            Hx = 1;
            Wx = 1;
            Ny = dy.dim(0);
            Cy = dy.dim(1);
            Hy = 1;
            Wy = 1;
            break;
        case 3:
            Nx = dx.dim(0);
            Cx = dx.dim(1);
            Hx = dx.dim(2);
            Wx = 1;
            Ny = dy.dim(0);
            Cy = dy.dim(1);
            Hy = dy.dim(2);
            Wy = 1;
            break;
        case 4:
            Nx = dx.dim(0);
            Cx = dx.dim(1);
            Hx = dx.dim(2);
            Wx = dx.dim(3);
            Ny = dy.dim(0);
            Cy = dy.dim(1);
            Hy = dy.dim(2);
            Wy = dy.dim(3);
            break;
        default:
            throw std::string("broadcasting only supports upto 4 dimensions.");
        }

        // zero
        math::set<T, CUDAContext>(dx.size(), T(0), dx_ptr);

        math::broadcast_gradient<T, CUDAContext>(dy_ptr, dx_ptr,
                                                 Ny, Cy, Hy, Wy,
                                                 Nx, Cx, Hx, Wx);

    }

private:
    Tensor dy;
    Tensor dx;
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
