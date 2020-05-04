#include "mlfe/core/op_algo.h"
#include "mlfe/device_context/cpu_context.h"

namespace mlfe{
namespace algorithm_cpu{

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

        // broadcast
        for(int i = 0; i < Ny; ++i){
            for(int j = 0; j < Cy; ++j){
                for(int k = 0; k < Hy; ++k){
                    for(int l = 0; l < Wy; ++l){
                        int x_idx = (i % Nx) * Cx * Hx * Wx +
                                    (j % Cx) * Hx * Wx +
                                    (k % Hx) * Wx +
                                    (l % Wx);
                        int y_idx = i * Cy * Hy * Wy +
                                    j * Hy * Wy +
                                    k * Wy +
                                    l;
                        y_ptr[y_idx] = x_ptr[x_idx];
                    }
                }
            }
        }
    }
private:
    Tensor x;
    Tensor y;
};

REGIST_OP_ALGO(Broadcasting)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CPU")
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
        std::fill(dx.begin<T>(), dx.end<T>(), 0);
        // accumulate gradient of parent.
        for(int i = 0; i < Ny; ++i){
            for(int j = 0; j < Cy; ++j){
                for(int k = 0; k < Hy; ++k){
                    for(int l = 0; l < Wy; ++l){
                        int dx_idx = (i % Nx) * Cx * Hx * Wx +
                                     (j % Cx) * Hx * Wx +
                                     (k % Hx) * Wx +
                                     (l % Wx);
                        int dy_idx = i * Cy * Hy * Wy +
                                     j * Hy * Wy +
                                     k * Wy +
                                     l;
                        dx_ptr[dx_idx] += dy_ptr[dy_idx];
                    }
                }
            }
        }
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
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = BroadcastingGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace cpu
} // end namespace mlfe
