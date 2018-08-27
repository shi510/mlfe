#include "../core/op_algo.h"
#include "../core/tensor_mem_ref.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../math/activations.h"
#include "../device_context/cuda_context.h"

namespace mlfe{ namespace algorithm_cuda{

template <class Dev, class Tp>
class ReduceMean : public OpAlgo{
using T = typename Tp::T;
public:
    ReduceMean(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        y = oac->GetVar("Y");
        size = x->Size();
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto y_ptr = y->Data<T>();
        math::set<T, CUDAContext>(y->Size(), T(0), y_ptr);
        math::sum<T, CUDAContext>(size, x_ptr, y_ptr);
        math::scal<T, CUDAContext>(1, T(1) / T(size), y_ptr, y_ptr);
    }
private:
    TensorMemRef *x;
    TensorMemRef *y;
    int size;
};

REGIST_OP_ALGO(ReduceMean)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReduceMean<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class ReduceMeanGrad : public OpAlgo{
using T = typename Tp::T;
public:
    ReduceMeanGrad(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        dy = oac->GetVar("dY");
        dx = oac->GetVar("dX");
        size = x->Size();
        scale = T(1) / T(size);
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto dy_ptr = dy->Data<T>();
        auto dx_ptr = dx->Data<T>();
        math::reduce_mean_gradient<T, CUDAContext>(size, scale, dy_ptr, dx_ptr);
    }

private:
    TensorMemRef *x;
    TensorMemRef *dy;
    TensorMemRef *dx;
    int size;
    T scale;
};

REGIST_OP_GRAD_ALGO(ReduceMean)
    .Input("X", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReduceMeanGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
