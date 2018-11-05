#include "../core/op_algo.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../device_context/cpu_context.h"
#include "../core/device.h"

namespace mlfe{
namespace algorithm_cpu{

template <class Tp>
class SquaredDifference : public OpAlgo{
using T = typename Tp::T;
public:
    SquaredDifference(OpAlgoContext *oac) : OpAlgo(oac, "SquaredDifference"){
        y = oac->get_output(0);
        x1 = y.get_children()[0];
        x2 = y.get_children()[1];
        size = x1.Size();
    }
    
    void Compute() override{
        auto x1_ptr = x1.device_data<T>();
        auto x2_ptr = x2.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        for(int n = 0; n < size; ++n){
            y_ptr[n] = std::pow(x1_ptr[n] - x2_ptr[n], 2);
        }
    }
private:
    Tensor x1;
    Tensor x2;
    Tensor y;
    int size;
};

REGIST_OP_ALGO(SquaredDifference)
    .Input("X1", type::float32::string)
    .Input("X2", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SquaredDifference<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class SquaredDifferenceGrad : public OpAlgo{
using T = typename Tp::T;
public:
    SquaredDifferenceGrad(OpAlgoContext *oac) : OpAlgo(oac){}

    void Compute() override{}

private:
    Tensor x1;
    Tensor x2;
    Tensor dy;
    Tensor dx1;
    Tensor dx2;
    int size;
};

REGIST_OP_GRAD_ALGO(SquaredDifference)
    .Input("X1", type::float32::string)
    .Input("X2", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX1", type::float32::string)
    .Output("dX2", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SquaredDifferenceGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
