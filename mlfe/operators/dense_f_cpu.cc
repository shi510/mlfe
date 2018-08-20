#include "../core/op_algo.h"
#include "../core/tensor_mem_ref.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../device_context/cpu_context.h"
#include "../core/device.h"

namespace mlfe{ namespace algorithm_cpu{

template <class Dev, class Tp>
class Dense : public OpAlgo{
using T = typename Tp::T;
public:
    Dense(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        w = oac->GetVar("W");
        b = oac->GetVar("B");
        y = oac->GetVar("Y");

        m = x->Shape()[0];
        n = w->Shape()[0];
        k = x->Shape()[1];

        bm = Device::Select<Dev>();

        bm.Allocate(m * Tp::size);

        math::set<T, CPUContext>(
            m,
            T(1),
            bm.Data<T>()
            );
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto w_ptr =w->Data<T>();
        auto b_ptr = b->Data<T>();
        auto y_ptr = y->Data<T>();
        auto bm_ptr = bm.Data<T>();

        /*
         * Forward computation.
         * y[batch_size, output_size] = 
         *   x[batch_size, input_size] * w[output_size, input_size]^T
         */
        math::gemm<T, CPUContext>(
            false, true,
            m, n, k,
            T(1), x_ptr, k,
            w_ptr, k,
            T(0), y_ptr, n, nullptr
            );

        /*
         * Add the bias.
         * y = y + b
         */
        math::gemm<T, CPUContext>(
            false, false,
            m, n, 1,
            T(1), bm_ptr, 1
            , b_ptr, n,
            T(1), y_ptr, n, nullptr
            );
    }
private:
    TensorMemRef *x;
    TensorMemRef *w;
    TensorMemRef *b;
    TensorMemRef *y;
    Device bm;
    int m, n, k;
};

REGIST_OP_ALGO(Dense)
    .Input("X", type::float32::string)
    .Input("W", type::float32::string)
    .Input("B", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CPU::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Dense<Device::CPU, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class DenseGrad : public OpAlgo{
using T = typename Tp::T;
public:
    DenseGrad(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        w = oac->GetVar("W");
        b = oac->GetVar("B");
        y = oac->GetVar("Y");
        dy = oac->GetVar("dY");
        dw = oac->GetVar("dW");
        db = oac->GetVar("dB");
        dx = oac->GetVar("dX");

        m = x->Shape()[0];
        n = w->Shape()[0];
        k = x->Shape()[1];

        bm = Device::Select<Dev>();

        bm.Allocate(m * Tp::size);

        math::set<T, CPUContext>(
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
        math::gemv<T, CPUContext>(
            true, m, n, T(1),
            dy_ptr, n,
            bm_ptr, T(0),
            db_ptr, n, nullptr);

        /*
         * gradient of weights.
         * dw[output_size, input_size] = 
         *   dy[batch_size, output_size]^T * x[batch_size, input_size]
         */
        math::gemm<T, CPUContext>(true, false,
            n, k, m,
            T(1), dy_ptr, n,
            x_ptr, k,
            T(0), dw_ptr, k, nullptr);

        /*
         * gradient of inputs.
         * dx[batch_size, input_size] = 
         *   dy[batch_size, output_size] * w[output_size, input_size] 
         */
        math::gemm<T, CPUContext>(
            false, false,
            m, k, n,
            T(1), dy_ptr, n,
            w_ptr, k,
            T(0), dx_ptr, k, nullptr);
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
    Device bm;
    int m, n, k;
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
    .Device(Device::CPU::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = DenseGrad<Device::CPU, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
