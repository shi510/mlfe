#include "mlfe/core/op_algo.h"
#include "mlfe/device_context/cpu_context.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/math/optimizers.h"

namespace mlfe{
namespace algorithm_cpu{

template <class Tp>
class Adam : public OpAlgo{
using T = typename Tp::T;
public:
    Adam(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_output(0);
        dx = x.grad();
        lr = oac->get_attr<T>("LearningRate");
        beta1 = oac->get_attr<T>("Beta1");
        beta2 = oac->get_attr<T>("Beta2");
        eps = oac->get_attr<T>("Epsilon");
        size = x.size();

        m_hist = create_memory(size * Tp::size);
        v_hist = create_memory(size * Tp::size);

        math::set<T, CPUContext>(
            size,
            static_cast<T>(0),
            m_hist->mutable_device_data<T>()
            );
        math::set<T, CPUContext>(
            size,
            static_cast<T>(0),
            v_hist->mutable_device_data<T>()
            );
    }

    void Compute() override{
        auto x_ptr = x.mutable_device_data<T>();
        auto dx_ptr = dx.device_data<T>();
        auto m_hist_ptr = m_hist->mutable_device_data<T>();
        auto v_hist_ptr = v_hist->mutable_device_data<T>();

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
    Tensor x;
    Tensor dx;
    Tensor y;
    memory_ptr m_hist;
    memory_ptr v_hist;
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
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Adam<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
