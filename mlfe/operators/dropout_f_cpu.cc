#include "mlfe/core/op_algo.h"
#include "mlfe/math/blas.h"
#include "mlfe/device_context/cpu_context.h"
#include "mlfe/core/device.h"

namespace mlfe{
namespace algorithm_cpu{

template <class Tp>
class Dropout : public OpAlgo{
using T = typename Tp::T;
public:
    Dropout(OpAlgoContext *oac) : OpAlgo(oac){
        y = oac->get_output(0);
        x = oac->get_input(0);
        mask = oac->get_attr<Tensor>("mask");
        keep_prob = oac->get_attr<Tensor>("keep_prob");
        resize();
    }

    void resize() override {
        mask.resize(x.shape());
        y.resize(x.shape());
        size = x.size();
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = x.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        auto mask_ptr = mask.mutable_device_data<T>();

        keep_ratio = keep_prob.data<T>()[0];
        keep_ratio_inv = T(1) / keep_ratio;
        if(keep_ratio != T(1)){
            b_dist = std::bernoulli_distribution(keep_ratio);
            for(int n = 0; n < size; ++n){
                T mask_val = mask_ptr[n] = b_dist(CPUContext::rng);
                y_ptr[n] = x_ptr[n] * mask_val * keep_ratio_inv;
            }
        }
        else{
            copy(x.get_memory(), y.get_memory());
        }
    }
private:
    Tensor x;
    Tensor y;
    Tensor mask;
    Tensor keep_prob;
    std::bernoulli_distribution b_dist;
    T keep_ratio, keep_ratio_inv;
    int size;
};

REGIST_OP_ALGO(Dropout)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Output("Mask", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Dropout<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class DropoutGrad : public OpAlgo{
using T = typename Tp::T;
public:
    DropoutGrad(OpAlgoContext *oac) : OpAlgo(oac){
        dx = oac->get_output(0);
        dy = oac->get_input(1);
        mask = oac->get_attr<Tensor>("mask");
        keep_prob = oac->get_attr<Tensor>("keep_prob");
        size = dy.size();
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();
        auto mask_ptr = mask.device_data<T>();

        keep_ratio = keep_prob.data<T>()[0];
        keep_ratio_inv = T(1) / (keep_ratio);
        if(keep_ratio != T(1)){
            for(int n = 0; n < size; ++n){
                dx_ptr[n] = dy_ptr[n] * mask_ptr[n] * keep_ratio_inv;
            }
        }
        else{
            for(int n = 0; n < size; ++n){
                dx_ptr[n] = dy_ptr[n];
            }
        }
    }

private:
    Tensor keep_prob;
    Tensor mask;
    Tensor dy;
    Tensor dx;
    T keep_ratio, keep_ratio_inv;
    int size;
};

REGIST_OP_GRAD_ALGO(Dropout)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("Mask", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = DropoutGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
