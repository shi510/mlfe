#include "../math/basic_functions.h"
#include "../core/op_algo.h"
#include "../core/tensor_mem_ref.h"
#include "../device_context/cpu_context.h"
#include <random>

namespace mlfe{

template <class Dev, class Tp>
class Constant : public OpAlgo{
using T = typename Tp::T;
public:
    Constant(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        value = oac->GetAttr<type::float32::T>("value");
        size = x->Size();
    }

    void Compute() override{
        math::set<T, CPUContext>(size, value, x->Data<T>());
    }
private:
    TensorMemRef *x;
    int size;
    type::float32::T value;
};

REGIST_OP_ALGO(Constant)
    .Input("X", type::float32::string)
    .Device(Device::CPU::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Constant<Device::CPU, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class Normal : public OpAlgo{
using T = typename Tp::T;
public:
    Normal(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        std = oac->GetAttr<type::float32::T>("std");
        clip = oac->GetAttr<bool>("clip");
        size = x->Size();
        dist = std::normal_distribution<T>(-std, std);
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
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
    TensorMemRef *x;
    type::float32::T std;
    bool clip;
    int size;
    std::normal_distribution<T> dist;
};

REGIST_OP_ALGO(Normal)
    .Input("X", type::float32::string)
    .Device(Device::CPU::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Normal<Device::CPU, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace mlfe
