#include "../core/op_algo.h"
#include "../device_context/cpu_context.h"
#include "../math/basic_functions.h"
#include "../math/optimizers.h"

namespace mlfe{ namespace optimizer{

template <class Dev, class Tp>
class AdaDelta : public OpAlgo{
using T = typename Tp::T;
public:
    AdaDelta(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        dx = oac->GetVar("dX");
        y = oac->GetVar("Y");
        lr = oac->GetAttr<T>("LearningRate");
        mr = oac->GetAttr<T>("MomentumRate");
        eps = oac->GetAttr<T>("Epsilon");
        size = x->Size();

        grad_hist = Device::Select<Dev>();
        grad_hist.Allocate(size * Tp::size);

        acc_hist = Device::Select<Dev>();
        acc_hist.Allocate(size * Tp::size);

        math::set<T, CPUContext>(
            size,
            static_cast<T>(0),
            grad_hist.Data<T>()
            );
        math::set<T, CPUContext>(
            size,
            static_cast<T>(0),
            acc_hist.Data<T>()
            );
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto dx_ptr = dx->Data<T>();
        auto y_ptr = y->Data<T>();
        auto grad_hist_ptr = grad_hist.Data<T>();
        auto acc_hist_ptr = acc_hist.Data<T>();

        math::adadelta<T, CPUContext>(
            size,
            x_ptr,
            dx_ptr,
            grad_hist_ptr,
            acc_hist_ptr,
            T(lr),
            T(mr),
            T(eps)
            );
    }

private:
    TensorMemRef *x;
    TensorMemRef *dx;
    TensorMemRef *y;
    Device grad_hist;
    Device acc_hist;
    int size;
    T lr;
    T mr;
    T eps;
};

REGIST_OP_ALGO(AdaDelta)
    .Input("X", type::float32::string)
    .Input("dX", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CPU::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = AdaDelta<Device::CPU, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace optimizer
} // end namespace mlfe
