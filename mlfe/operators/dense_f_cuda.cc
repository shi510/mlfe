#include "../core/op_algo.h"
#include "../core/tensor_mem_ref.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../device_context/cuda_context.h"
#include "../core/device.h"

namespace mlfe{ namespace algorithm_cuda{

template <class Dev, class Tp>
class Dense : public OpAlgo{
using T = typename Tp::T;
public:
    Dense(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_input(0);
        w = oac->get_input(1);
        b = oac->get_input(2);
        y = oac->get_output(0);

        m = x->Shape()[0];
        n = w->Shape()[0];
        k = x->Shape()[1];

        bm = oac->GetDevice().CreateDeviceMemory();
        bm.Allocate(m * Tp::size);

        math::set<T, CUDAContext>(
            m,
            T(1),
            bm.Data<T>()
            );
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto w_ptr = w->Data<T>();
        auto b_ptr = b->Data<T>();
        auto y_ptr = y->Data<T>();
        auto bm_ptr = bm.Data<T>();

        /*
         * Forward computation.
         * y[batch_size, output_size] =
         *   x[batch_size, input_size] * w[output_size, input_size]^T
         */
        math::gemm<Tp::T, CUDAContext>(
            false, true,
            m, n, k,
            Tp::T(1), x_ptr, k,
            w_ptr, k,
            Tp::T(0), y_ptr, n, &cuda
            );

        /*
         * Add the bias.
         * y = y + b
         */
        math::gemm<Tp::T, CUDAContext>(
            false, false,
            m, n, 1,
            Tp::T(1), bm_ptr, 1
            , b_ptr, n,
            Tp::T(1), y_ptr, n, &cuda
            );
    }
private:
    TensorMemRef *x;
    TensorMemRef *w;
    TensorMemRef *b;
    TensorMemRef *y;
    DeviceMemory bm;
    int m, n, k;
    CUDAContext cuda;
};

REGIST_OP_ALGO(Dense)
    .Input("X", type::float32::string)
    .Input("W", type::float32::string)
    .Input("B", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Dense<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class DenseGrad : public OpAlgo{
using T = typename Tp::T;
public:
    DenseGrad(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_input(0);
        w = oac->get_input(1);
        b = oac->get_input(2);
        y = oac->get_input(3);
        dy = oac->get_input(4);
        dw = oac->get_output(0);
        db = oac->get_output(1);
        dx = oac->get_output(2);
        m = x->Shape()[0];
        n = w->Shape()[0];
        k = x->Shape()[1];

        bm = oac->GetDevice().CreateDeviceMemory();
        bm.Allocate(m * Tp::size);

        math::set<T, CUDAContext>(
            m,
            T(1),
            bm.Data<T>()
            );
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto w_ptr = w->Data<T>();
        auto b_ptr = b->Data<T>();
        auto y_ptr = y->Data<T>();
        auto dy_ptr = dy->Data<T>();
        auto dw_ptr = dw->Data<T>();
        auto db_ptr = db->Data<T>();
        auto dx_ptr = dx->Data<T>();
        auto bm_ptr = bm.Data<T>();

        /* gradient of biases.
         * db = dy.
         */
        math::gemv<T, CUDAContext>(
            true, m, n, T(1),
            dy_ptr, n,
            bm_ptr, T(0),
            db_ptr, n, &cuda);

        /*
         * gradient of weights.
         * dw[output_size, input_size] =
         *   dy[batch_size, output_size]^T * x[batch_size, input_size]
         */
        math::gemm<T, CUDAContext>(true, false,
            n, k, m,
            T(1), dy_ptr, n,
            x_ptr, k,
            T(0), dw_ptr, k, &cuda);

        /*
         * gradient of inputs.
         * dx[batch_size, input_size] =
         *   dy[batch_size, output_size] * w[output_size, input_size]
         */
        math::gemm<T, CUDAContext>(
            false, false,
            m, k, n,
            T(1), dy_ptr, n,
            w_ptr, k,
            T(0), dx_ptr, k, &cuda);
    }

private:
    TensorMemRef *x;
    TensorMemRef *w;
    TensorMemRef *b;
    TensorMemRef *y;
    TensorMemRef *dy;
    TensorMemRef *dw;
    TensorMemRef *db;
    TensorMemRef *dx;
    DeviceMemory bm;
    int m, n, k;
    CUDAContext cuda;
};

REGIST_OP_GRAD_ALGO(Dense)
    .Input("X", type::float32::string)
    .Input("W", type::float32::string)
    .Input("B", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dW", type::float32::string)
    .Output("dB", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = DenseGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
