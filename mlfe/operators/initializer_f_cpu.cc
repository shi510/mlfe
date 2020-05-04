#include "../math/basic_functions.h"
#include "../core/op_algo.h"
#include "../device_context/cpu_context.h"
#include <random>

namespace mlfe{
namespace algorithm_cpu{

template <class Tp>
class Constant : public OpAlgo{
using T = typename Tp::T;
public:
    Constant(OpAlgoContext *oac) : OpAlgo(oac, "Constant"){
        y = oac->get_output(0);
        value = oac->get_attr<float>("value");
        size = y.size();
    }

    void Compute(op_algo_runtime_context& rc) override{
        math::set<T, CPUContext>(size, value, y.mutable_device_data<T>());
    }

private:
    Tensor y;
    int size;
    type::float32::T value;
};

REGIST_OP_ALGO(Constant)
    .Input("X", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Constant<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class Normal : public OpAlgo{
using T = typename Tp::T;
public:
    Normal(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_output(0);
        std = oac->get_attr<type::float32::T>("std");
        clip = oac->get_attr<bool>("clip");
        size = x.size();
        dist = std::normal_distribution<T>(-std, std);
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = x.mutable_device_data<T>();
        for(int n = 0; n < size; ++n){
            x_ptr[n] = dist(CPUContext::rng);
        }
        if(clip){
            for(int n = 0; n < size; ++n){
                if(x_ptr[n] < -std){
                    x_ptr[n] = -std;
                }
                else if(x_ptr[n] > std){
                    x_ptr[n] = std;
                }
            }
        }
    }

private:
    Tensor x;
    type::float32::T std;
    bool clip;
    int size;
    std::normal_distribution<T> dist;
};

REGIST_OP_ALGO(Normal)
    .Output("X", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Normal<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
