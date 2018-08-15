#include "../core/op_algo.h"
#include "../core/tensor_mem_ref.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../device_context/cuda_context.h"
#include "../core/device.h"

namespace mlfe{

template <class Dev, class Tp>
class Dropout : public OpAlgo{
using T = typename Tp::T;
public:
    Dropout(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        y = oac->GetVar("Y");
        mask = oac->GetVar("Mask");
        drop_ratio = oac->GetAttr<T>("dropout_ratio");
        training = oac->GetAttr<bool>("is_training_step");
        drop_ratio_inv = T(1) / (T(1) - drop_ratio);
        size = x->Size();
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto y_ptr = y->Data<T>();
        auto mask_ptr = mask->Data<T>();
        if(training){
            math::bernoulli_distribution<T, CUDAContext>(size, drop_ratio, mask_ptr);
            math::scal<T, CUDAContext>(size, drop_ratio_inv, mask_ptr, y_ptr);
            math::elementwise_mul<T, CUDAContext>(size, x_ptr, y_ptr, y_ptr);
        }
        else{
            Device::Copy<Device::CUDA, Device::CUDA>(x->GetDevice(), y->GetDevice());
        }
    }
private:
    TensorMemRef *x;
    TensorMemRef *y;
    TensorMemRef *mask;
    T drop_ratio, drop_ratio_inv;
    bool training;
    int size;
};

REGIST_OP_ALGO(Dropout)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Output("Mask", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Dropout<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class DropoutGrad : public OpAlgo{
using T = typename Tp::T;
public:
    DropoutGrad(OpAlgoContext *oac) : OpAlgo(oac){
        dy = oac->GetVar("dY");
        dx = oac->GetVar("dX");
        mask = oac->GetVar("Mask");
        drop_ratio = oac->GetAttr<T>("dropout_ratio");
        drop_ratio_inv = T(1) / (T(1) - drop_ratio);
        size = dy->Size();
    }

    void Compute() override{
        auto dy_ptr = dy->Data<T>();
        auto dx_ptr = dx->Data<T>();
        auto mask_ptr = mask->Data<T>();
        math::scal<T, CUDAContext>(size, drop_ratio_inv, mask_ptr, dx_ptr);
        math::elementwise_mul<T, CUDAContext>(size, dy_ptr, dx_ptr, dx_ptr);
    }

private:
    TensorMemRef *x;
    TensorMemRef *y;
    TensorMemRef *dy;
    TensorMemRef *dx;
    TensorMemRef *mask;
    T drop_ratio, drop_ratio_inv;
    int size;
};

REGIST_OP_GRAD_ALGO(Dropout)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("Mask", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = DropoutGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace mlfe
