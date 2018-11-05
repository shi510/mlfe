#include "../core/op_algo.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../math/activations.h"
#include "../device_context/cuda_context.h"
#include <iostream>

namespace mlfe{
namespace algorithm_cuda{

template <class Tp>
class ReduceMean : public OpAlgo{
using T = typename Tp::T;
public:
    ReduceMean(OpAlgoContext *oac) : OpAlgo(oac, "ReduceMean"){
        y = oac->get_output(0);
        x = y.get_children()[0];
        size = x.Size();
    }

    void Compute() override{
        auto x_ptr = x.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        math::set<T, CUDAContext>(1, T(0), y_ptr);
        math::sum<T, CUDAContext>(size, x_ptr, y_ptr);
        math::scal<T, CUDAContext>(1, T(1) / T(size), y_ptr, y_ptr);
    }

private:
    Tensor x;
    Tensor y;
    int size;
};

REGIST_OP_ALGO(ReduceMean)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReduceMean<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class ReduceMeanGrad : public OpAlgo{
using T = typename Tp::T;
public:
    ReduceMeanGrad(OpAlgoContext *oac) : OpAlgo(oac, "ReduceMeanGradient"){
        dx = oac->get_output(0);
        dy = dx.get_children()[0];
        size = dx.Size();
        scale = T(1) / T(size);
    }

    void Compute() override{
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();
        math::reduce_mean_gradient<T, CUDAContext>(size, 
                                                   scale, 
                                                   dy_ptr, 
                                                   dx_ptr
                                                  );
    }

private:
    Tensor dy;
    Tensor dx;
    int size;
    T scale;
};

REGIST_OP_GRAD_ALGO(ReduceMean)
    .Input("X", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReduceMeanGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
