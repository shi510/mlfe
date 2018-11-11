#include "../core/op_algo.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../device_context/cuda_context.h"
#include "../math/basic_functions.h"
#include <curand.h>

namespace mlfe{
namespace algorithm_cuda{

template <class Tp>
class Constant : public OpAlgo{
using T = typename Tp::T;
public:
    Constant(OpAlgoContext *oac) : OpAlgo(oac, "Constant"){
        y = oac->get_output(0);
        value = oac->get_attr<type::float32::T>("value");
        size = y.size();
    }

    void Compute() override{
        math::set<T, CUDAContext>(size,
                                  value, 
                                  y.mutable_device_data<T>()
                                 );
    }

private:
    Tensor y;
    int size;
    type::float32::T value;
};

REGIST_OP_ALGO(Constant)
    .Output("Y", type::float32::string)
    .Device("CUDA")
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
        y = oac->get_output(0);
        std = oac->get_attr<type::float32::T>("std");
        clip = oac->get_attr<bool>("clip");
        size = y.size();
    }

    void Compute() override{
        auto y_ptr = y.mutable_device_data<T>();
        curandGenerateNormal(CUDAContext::rng, y_ptr, size, 0, std);
        if(clip){
            math::clip_min_max<T, CUDAContext>(size, y_ptr, -std, std);
        }
    }

private:
    Tensor y;
    type::float32::T std;
    bool clip;
    int size;
};

REGIST_OP_ALGO(Normal)
    .Input("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Normal<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
