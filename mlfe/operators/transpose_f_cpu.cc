#include "mlfe/core/op_algo.h"
#include "mlfe/core/device.h"
#include <stdexcept>
#include <numeric>
#include <hptt.h>

namespace mlfe{
namespace algorithm_cpu{

template <class Tp>
class Transposition : public OpAlgo{
using T = typename Tp::T;
public:
    Transposition(OpAlgoContext *oac) : OpAlgo(oac, "Transposition"){
        x = oac->get_input(0);
        perm = oac->get_input(1);
        y = oac->get_output(0);
        resize();
    }

    void resize() override {
        x_shape = x.shape();
        auto y_shape = std::vector<int>(x.dims());
        for(int n = 0; n < x.dims(); ++n){
            y_shape[n] = x_shape[perm.data<int>()[n]];
        }
        y.resize(y_shape);
        plan = hptt::create_plan(perm.data<int>(), x.dims(), 
            T(1), x.device_data<T>(), x_shape.data(), NULL, 
            T(0), y.mutable_device_data<T>(), NULL, 
            hptt::ESTIMATE, 1, nullptr, true);
    }

    void Compute(op_algo_runtime_context& rc) override{
        plan->execute();
    }

private:
    Tensor x;
    Tensor y;
    Tensor perm;
    std::vector<int> x_shape;
    std::shared_ptr<hptt::Transpose<T>> plan;
};

REGIST_OP_ALGO(Transposition)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Transposition<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class TranspositionGrad : public OpAlgo{
using T = typename Tp::T;
public:
    TranspositionGrad(OpAlgoContext *oac)
        : OpAlgo(oac, "TranspositionGradient"){
        x = oac->get_input(0);
        perm = oac->get_input(1);
        dy = oac->get_input(2);
        dx = oac->get_output(0);
        resize();
    }

    void resize() override {
        dy_shape = dy.shape();
        auto dx_shape = std::vector<int>(x.dims());
        auto re_perm = std::vector<int>(x.dims());
        for(int n = 0; n < x.dims(); ++n){
            re_perm[perm.data<int>()[n]] = n;
        }
        for(int n = 0; n < x.dims(); ++n){
            dx_shape[n] = dy_shape[re_perm[n]];
        }
        dx.resize(dx_shape);
        plan = hptt::create_plan(re_perm.data(), x.dims(), 
            T(1), dy.device_data<T>(), dy_shape.data(), NULL, 
            T(0), dx.mutable_device_data<T>(), NULL, 
            hptt::ESTIMATE, 1, nullptr, true);
    }

    void Compute(op_algo_runtime_context& rc) override{
        plan->execute();
    }

private:
    Tensor x;
    Tensor dy;
    Tensor dx;
    Tensor perm;
    std::vector<int> dy_shape;
    std::shared_ptr<hptt::Transpose<T>> plan;
};

REGIST_OP_GRAD_ALGO(Transposition)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = TranspositionGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
