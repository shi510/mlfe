#include "../core/op_algo.h"
#include "../core/tensor_mem_ref.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../device_context/cuda_context.h"
#include "../core/device.h"

namespace mlfe{
namespace algorithm_cuda{

template <class Dev, class Tp>
class SquaredDifference : public OpAlgo{
using T = typename Tp::T;
public:
    SquaredDifference(OpAlgoContext *oac) : OpAlgo(oac){
        x1 = oac->get_input(0);
        x2 = oac->get_input(1);
        y = oac->get_output(0);
        size = x1->Size();
    }

    void Compute() override{
        auto x1_ptr = x1->Data<T>();
        auto x2_ptr = x2->Data<T>();
        auto y_ptr = y->Data<T>();
        
        math::squared_difference<T, CUDAContext>(size, x1_ptr, x2_ptr, y_ptr);
    }

private:
    TensorMemRef *x1;
    TensorMemRef *x2;
    TensorMemRef *y;
    int size;
};

REGIST_OP_ALGO(SquaredDifference)
    .Input("X1", type::float32::string)
    .Input("X2", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SquaredDifference<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class SquaredDifferenceGrad : public OpAlgo{
using T = typename Tp::T;
public:
    SquaredDifferenceGrad(OpAlgoContext *oac) : OpAlgo(oac){
        x1 = oac->get_input(0);
        x2 = oac->get_input(1);
        dy = oac->get_input(2);
        dx1 = oac->get_output(0);
        dx2 = oac->get_output(1);
        size = x1->Size();
    }

    void Compute() override{
        auto x1_ptr = x1->Data<T>();
        auto x2_ptr = x2->Data<T>();
        auto dy_ptr = dy->Data<T>();
        auto dx1_ptr = dx1->Data<T>();
        auto dx2_ptr = dx2->Data<T>();
        
        math::SubCuda(size, x1_ptr, x2_ptr, dx1_ptr);
        math::MulValCuda(size, T(2), dx1_ptr, dx1_ptr);
        math::MulCuda(size, dy_ptr, dx1_ptr, dx1_ptr);
        Device::Copy<Device::CUDA, Device::CUDA>(dx1->GetDeviceMemory(), dx2->GetDeviceMemory());
    }

private:
    TensorMemRef *x1;
    TensorMemRef *x2;
    TensorMemRef *dy;
    TensorMemRef *dx1;
    TensorMemRef *dx2;
    int size;
};

REGIST_OP_GRAD_ALGO(SquaredDifference)
    .Input("X1", type::float32::string)
    .Input("X2", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX1", type::float32::string)
    .Output("dX2", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SquaredDifferenceGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
