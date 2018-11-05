#include "../core/op_algo.h"
#include "../device_context/cpu_context.h"

namespace mlfe{
namespace algorithm_cpu{

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
        T sum = T(0);
        for(int n = 0; n < size; ++n){
            sum += x_ptr[n];
        }
        y_ptr[0] = sum / T(size);
    }
private:
    Tensor x;
    Tensor y;
    int size;
};

REGIST_OP_ALGO(ReduceMean)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CPU")
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
        T dy_val = dy_ptr[0];

        for(int n = 0; n < size; ++n){
            dx_ptr[n] = dy_val * scale;
        }
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
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReduceMeanGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
