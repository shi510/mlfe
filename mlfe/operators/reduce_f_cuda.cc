#include "../core/op_algo.h"
#include "../core/tensor_mem_ref.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../math/activations.h"
#include "../device_context/cuda_context.h"

namespace mlfe{

template <class Dev, class Tp>
class ReduceMean : public OpAlgo{
using T = typename Tp::T;
public:
    ReduceMean(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        y = oac->GetVar("Y");
        size = x->Size();
        reduce_buf = Device::Select<Device::CUDA>();
        reduce_buf.Allocate(CUDA_CONTEXT_GET_BLOCKS(size) * Tp::size);
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto y_ptr = y->Data<T>();
        auto reduce_ptr = reduce_buf.Data<T>();
        math::set<T, CUDAContext>(reduce_buf.Size(), T(0), reduce_ptr);
        math::sum<T, CUDAContext>(size, x_ptr, reduce_ptr);
        math::sum<T, CUDAContext>(CUDA_CONTEXT_GET_BLOCKS(size), reduce_ptr, y_ptr);
        math::scal<T, CUDAContext>(1, T(1) / T(size), y_ptr, y_ptr);
    }
private:
    TensorMemRef *x;
    TensorMemRef *y;
    Device reduce_buf;
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
        dy_d = Device::Select<Device::CPU>();
        dx_d = Device::Select<Device::CPU>();
        dy_d.Allocate(dy->Size() * Tp::size);
        dx_d.Allocate(dx->Size() * Tp::size);
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto dy_ptr = dy->Data<T>();
        auto dx_ptr = dx->Data<T>();
        //Device::Copy<Device::CUDA, Device::CPU>(dy->GetDevice(), dy_d);
        math::reduce_mean_gradient<T, CUDAContext>(size, scale, dy_ptr, dx_ptr);
         //for(int n = 0; n < size; ++n){
         //    dx_d.Data<T>()[n] = dy_d.Data<T>()[0] * scale;
         //}
        //Device::Copy<Device::CPU, Device::CUDA>(dx_d, dx->GetDevice());
    }

private:
    TensorMemRef *x;
    TensorMemRef *dy;
    TensorMemRef *dx;
    Device dy_d, dx_d;
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

} // end namespace mlfe
