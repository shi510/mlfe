#include "mlfe/core/op_algo.h"
#include "mlfe/device_context/cpu_context.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/math/optimizers.h"

namespace mlfe{
namespace algorithm_cpu{

template <class Tp>
class GradientDescent : public OpAlgo{
using T = typename Tp::T;
public:
    GradientDescent(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_output(0);
        x_grad = x.grad();
        lr = oac->get_attr<Tensor>("LearningRate");
        mr = oac->get_attr<T>("MomentumRate");
        wd = oac->get_attr<T>("WeightDecay");
        size = x.size();
        mmt_hist = create_memory(size * Tp::size);

        math::set<T, CPUContext>(
            size,
            static_cast<T>(0),
            mmt_hist->mutable_device_data<T>()
            );
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = x.mutable_device_data<T>();
        auto dx_ptr = x_grad.device_data<T>();
        auto mmt_hist_ptr = mmt_hist->mutable_device_data<T>();

        math::gradient_descent_momentum<float, CPUContext>(
            size,
            x_ptr,
            dx_ptr,
            mmt_hist_ptr,
            static_cast<float>(lr.data<T>()[0]),
            static_cast<float>(mr),
            static_cast<float>(wd)
            );
    }

private:
    Tensor x;
    Tensor x_grad;
    Tensor lr;
    memory_ptr mmt_hist;
    int size;
    T mr;
    T wd;
};

REGIST_OP_ALGO(GradientDescent)
    .Input("X", type::float32::string)
    .Input("dX", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = GradientDescent<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
