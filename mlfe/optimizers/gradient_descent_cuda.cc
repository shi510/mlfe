#include "../core/op_algo.h"
#include "../device_context/cuda_context.h"
#include "../math/basic_functions.h"
#include "../math/optimizers.h"
#include "../math/blas.h"

namespace mlfe{
namespace algorithm_cuda{

template <class Tp>
class GradientDescent : public OpAlgo{
using T = typename Tp::T;
public:
    GradientDescent(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_output(0);
        x_grad = x.grad();
        lr = oac->get_attr<T>("LearningRate");
        size = x.size();
    }

    void Compute() override{
        auto x_ptr = x.mutable_device_data<T>();
        auto dx_ptr = x_grad.device_data<T>();

        // X = X - LearningRate*dX
        math::axpy<T, CUDAContext>(size,
                                   -lr,
                                   dx_ptr,
                                   x_ptr
                                  );
    }

private:
    Tensor x;
    Tensor x_grad;
    int size;
    T lr;
};

REGIST_OP_ALGO(GradientDescent)
    .Input("X", type::float32::string)
    .Input("dX", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = GradientDescent<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class GradientDescentWithMomentum : public OpAlgo{
using T = typename Tp::T;
public:
    GradientDescentWithMomentum(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_output(0);
        x_grad = x.grad();
        lr = oac->get_attr<T>("LearningRate");
        mr = oac->get_attr<T>("MomentumRate");
        wd = oac->get_attr<T>("WeightDecay");
        size = x.size();
        mmt_hist = create_memory(size * Tp::size);

        math::set<T, CUDAContext>(
            size,
            static_cast<T>(0),
            mmt_hist->mutable_device_data<T>()
            );
    }

    void Compute() override{
        auto x_ptr = x.mutable_device_data<T>();
        auto dx_ptr = x_grad.device_data<T>();
        auto mmt_hist_ptr = mmt_hist->mutable_device_data<T>();

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
    Tensor x;
    Tensor x_grad;
    memory_ptr mmt_hist;
    int size;
    T lr;
    T mr;
    T wd;
};

REGIST_OP_ALGO(GradientDescentWithMomentum)
    .Input("X", type::float32::string)
    .Input("dX", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = GradientDescentWithMomentum<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
