#include "../core/op_algo.h"
#include "../device_context/cuda_context.h"
#include "../math/basic_functions.h"
#include "../math/optimizers.h"
#include "../math/blas.h"

namespace mlfe{ namespace optimizer{

template <class Dev, class Tp>
class GradientDescent : public OpAlgo{
using T = typename Tp::T;
public:
    GradientDescent(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        dx = oac->GetVar("dX");
        y = oac->GetVar("Y");
        lr = oac->GetAttr<T>("LearningRate");
        size = x->Size();
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto dx_ptr = dx->Data<T>();
        auto y_ptr = y->Data<T>();

        // X = X - LearningRate*dX
        math::axpy<T, CUDAContext>(
            size,
            -lr,
            dx_ptr,
            y_ptr
            );
    }

private:
    TensorMemRef *x;
    TensorMemRef *dx;
    TensorMemRef *y;
    int size;
    T lr;
};

REGIST_OP_ALGO(GradientDescent)
    .Input("X", type::float32::string)
    .Input("dX", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = GradientDescent<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class GradientDescentWithMomentum : public OpAlgo{
using T = typename Tp::T;
public:
    GradientDescentWithMomentum(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        dx = oac->GetVar("dX");
        y = oac->GetVar("Y");
        lr = oac->GetAttr<T>("LearningRate");
        mr = oac->GetAttr<T>("MomentumRate");
        wd = oac->GetAttr<T>("WeightDecay");
        size = x->Size();
        mmt_hist = Device::Select<Dev>();
        mmt_hist.Allocate(size * Tp::size);

        math::set<T, CUDAContext>(
            size,
            static_cast<T>(0),
            mmt_hist.Data<T>()
            );
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto dx_ptr = dx->Data<T>();
        auto y_ptr = y->Data<T>();
        auto mmt_hist_ptr = mmt_hist.Data<T>();

        math::gradient_descent_momentum<float, CUDAContext>(
            size,
            x_ptr,
            dx_ptr,
            mmt_hist_ptr,
            static_cast<float>(lr),
            static_cast<float>(mr),
            static_cast<float>(wd)
            );
    }

private:
    TensorMemRef *x;
    TensorMemRef *dx;
    TensorMemRef *y;
    Device mmt_hist;
    int size;
    T lr;
    T mr;
    T wd;
};

REGIST_OP_ALGO(GradientDescentWithMomentum)
    .Input("X", type::float32::string)
    .Input("dX", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = GradientDescentWithMomentum<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace optimizer
} // end namespace mlfe
