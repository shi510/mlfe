#include "../core/op_algo.h"
#include "../core/device.h"
#include "../core/tensor_mem_ref.h"

namespace mlfe{

template <class Dev, class Tp>
class Reshape : public OpAlgo{
using T = typename Tp::T;
public:
    Reshape(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        y = oac->GetVar("Y");
    }

    // TODO : Do not use Copy.
    void Compute() override{
        Device::Copy<Device::CUDA, Device::CUDA>(x->GetDevice(), y->GetDevice());
    }
private:
    TensorMemRef *x;
    TensorMemRef *y;
};

REGIST_OP_ALGO(Reshape)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Reshape<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class ReshapeGrad : public OpAlgo{
using T = typename Tp::T;
public:
    ReshapeGrad(OpAlgoContext *oac) : OpAlgo(oac){
        dy = oac->GetVar("dY");
        dx = oac->GetVar("dX");
    }

    // TODO : Do not use Copy.
    void Compute() override{
        Device::Copy<Device::CUDA, Device::CUDA>(dy->GetDevice(), dx->GetDevice());
    }

private:
    TensorMemRef *dy;
    TensorMemRef *dx;
};

REGIST_OP_GRAD_ALGO(Reshape)
    .Input("X", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReshapeGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace mlfe
