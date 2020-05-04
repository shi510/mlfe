#include "mlfe/core/op_algo.h"
#include "mlfe/device_context/cuda_context.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/math/optimizers.h"
#include "mlfe/math/blas.h"

namespace mlfe{
namespace algorithm_cuda{

template <class Tp>
class AdaDelta : public OpAlgo{
using T = typename Tp::T;
public:
    AdaDelta(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_output(0);
        dx = x.grad();
        lr = oac->get_attr<T>("LearningRate");
        mr = oac->get_attr<T>("MomentumRate");
        eps = oac->get_attr<T>("Epsilon");
        size = x.size();

        grad_hist = create_memory(size * Tp::size);
        acc_hist = create_memory(size * Tp::size);

        math::set<T, CUDAContext>(
            size,
            static_cast<T>(0),
            grad_hist->mutable_device_data<T>()
            );
        math::set<T, CUDAContext>(
            size,
            static_cast<T>(0),
            acc_hist->mutable_device_data<T>()
            );
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = x.mutable_device_data<T>();
        auto dx_ptr = dx.device_data<T>();
        auto grad_hist_ptr = grad_hist->mutable_device_data<T>();
        auto acc_hist_ptr = acc_hist->mutable_device_data<T>();

        math::adadelta<T, CUDAContext>(
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
    Tensor x;
    Tensor dx;
    Tensor y;
    memory_ptr grad_hist;
    memory_ptr acc_hist;
    int size;
    T lr;
    T mr;
    T eps;
};

REGIST_OP_ALGO(AdaDelta)
    .Input("X", type::float32::string)
    .Input("dX", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = AdaDelta<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
