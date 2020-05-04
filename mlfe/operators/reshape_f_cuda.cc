#include "../core/op_algo.h"
#include "../core/device.h"

namespace mlfe{
namespace algorithm_cuda{

template <class Tp>
class Reshape : public OpAlgo{
using T = typename Tp::T;
public:
    Reshape(OpAlgoContext *oac) : OpAlgo(oac, "Reshape"){}

    void Compute(op_algo_runtime_context& rc) override{}
private:
    Tensor x;
    Tensor y;
};

REGIST_OP_ALGO(Reshape)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Reshape<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class ReshapeGrad : public OpAlgo{
using T = typename Tp::T;
public:
    ReshapeGrad(OpAlgoContext *oac) 
        : OpAlgo(oac, "ReshapeGradient"){}

    void Compute(op_algo_runtime_context& rc) override{}

private:
    Tensor dy;
    Tensor dx;
};

REGIST_OP_GRAD_ALGO(Reshape)
    .Input("X", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReshapeGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
