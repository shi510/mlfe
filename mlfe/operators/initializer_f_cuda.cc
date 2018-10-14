#include "../core/op_algo.h"
#include "../core/tensor_mem_ref.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../device_context/cuda_context.h"
#include "../math/basic_functions.h"
#include <curand.h>

namespace mlfe{

template <class Dev, class Tp>
class Constant : public OpAlgo{
using T = typename Tp::T;
public:
    Constant(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_input(0);
        value = oac->GetAttr<type::float32::T>("value");
        size = x->Size();
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        math::set<T, CUDAContext>(size, value, x_ptr);
    }
private:
    TensorMemRef *x;
    int size;
    type::float32::T value;
};

REGIST_OP_ALGO(Constant)
    .Input("X", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Constant<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class Normal : public OpAlgo{
using T = typename Tp::T;
public:
    Normal(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_input(0);
        std = oac->GetAttr<type::float32::T>("std");
        clip = oac->GetAttr<bool>("clip");
        size = x->Size();
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        curandGenerateNormal(CUDAContext::rng, x_ptr, size, 0, std);
        if(clip){
            math::clip_min_max<T, CUDAContext>(size, x_ptr, -std, std);
        }
    }

private:
    TensorMemRef *x;
    type::float32::T std;
    bool clip;
    int size;
};

REGIST_OP_ALGO(Normal)
    .Input("X", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Normal<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class Xavier : public OpAlgo{
using T = typename Tp::T;
public:
    Xavier(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_input(0);
        a = oac->GetAttr<type::int32::T>("a");
        b = oac->GetAttr<type::int32::T>("b");
        size = x->Size();
        var = static_cast<T>(2) / static_cast<T>(a);
        trunc_val = std::sqrt(
            static_cast<T>(6) / static_cast<T>(a + b)
        );
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        curandGenerateUniform(CUDAContext::rng, x_ptr, size);
        math::shift_a_b<T, CUDAContext>(size, x_ptr, -var, var);
        //math::UniformCurand<T>(&CUDAContext::rng, size, x_ptr, -var, var);
        math::clip_min_max<T, CUDAContext>(size, x_ptr, -trunc_val, trunc_val);
    }

private:
    TensorMemRef *x;
    int size;
    type::int32::T a;
    type::int32::T b;
    T var;
    T trunc_val;
};

REGIST_OP_ALGO(Xavier)
    .Input("X", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Xavier<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace mlfe
