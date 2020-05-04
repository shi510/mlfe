#include "../core/op_algo.h"
#include "../core/device.h"

namespace mlfe{
namespace algorithm_cpu{

template <class Tp>
class Reshape : public OpAlgo{
    using T = typename Tp::T;
public:
    Reshape(OpAlgoContext *oac) : OpAlgo(oac, "Reshape"){
        //x = oac->get_input(0);
        //y = oac->get_output(0);
    }

    // TODO : Do not use Copy.
    void Compute(op_algo_runtime_context& rc) override{
        //Device::Copy<Device::CPU>(x->GetDeviceMemory(), y->GetDeviceMemory());
    }

private:
    Tensor x;
    Tensor y;
};

REGIST_OP_ALGO(Reshape)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Reshape<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class ReshapeGrad : public OpAlgo{
using T = typename Tp::T;
public:
    ReshapeGrad(OpAlgoContext *oac) : OpAlgo(oac){
        //dy = oac->get_input(1);
        //dx = oac->get_output(0);
    }

    // TODO : Do not use Copy.
    void Compute(op_algo_runtime_context& rc) override{
        //Device::Copy<Device::CPU>(dy->GetDeviceMemory(), dx->GetDeviceMemory());
    }

private:
    Tensor dy;
    Tensor dx;
};

REGIST_OP_GRAD_ALGO(Reshape)
    .Input("X", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReshapeGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
