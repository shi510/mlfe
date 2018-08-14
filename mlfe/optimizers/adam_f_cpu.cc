#include "../core/op_algo.h"
#include "../device_context/cpu_context.h"
#include "../math/basic_functions.h"
#include "../math/optimizers.h"

namespace mlfe{ namespace optimizer{

template <class Dev, class Tp>
class Adam : public OpAlgo{
using T = typename Tp::T;
public:
    Adam(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        dx = oac->GetVar("dX");
        y = oac->GetVar("Y");
        lr = oac->GetAttr<T>("LearningRate");
        beta1 = oac->GetAttr<T>("Beta1");
        beta2 = oac->GetAttr<T>("Beta2");
        eps = oac->GetAttr<T>("Epsilon");
        size = x->Size();

        m_hist = Device::Select<Dev>();
        m_hist.Allocate(size * Tp::size);

        v_hist = Device::Select<Dev>();
        v_hist.Allocate(size * Tp::size);

        math::set<T, CPUContext>(
            size,
            static_cast<T>(0),
            m_hist.Data<T>()
            );
        math::set<T, CPUContext>(
            size,
            static_cast<T>(0),
            v_hist.Data<T>()
            );
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto dx_ptr = dx->Data<T>();
        auto y_ptr = y->Data<T>();
        auto m_hist_ptr = m_hist.Data<T>();
        auto v_hist_ptr = v_hist.Data<T>();

        math::adam<T, CPUContext>(
            size,
            x_ptr,
            dx_ptr,
            m_hist_ptr,
            v_hist_ptr,
            T(lr),
            T(beta1),
            T(beta2),
            T(eps)
            );
    }

private:
    TensorMemRef *x;
    TensorMemRef *dx;
    TensorMemRef *y;
    Device m_hist;
    Device v_hist;
    int size;
    T lr;
    T beta1;
    T beta2;
    T eps;
};

REGIST_OP_ALGO(Adam)
    .Input("X", type::float32::string)
    .Input("dX", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CPU::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Adam<Device::CPU, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace optimizer
} // end namespace mlfe
