#include "../core/op_algo.h"
#include "../core/tensor_mem_ref.h"
#include "../device_context/cpu_context.h"

namespace mlfe{

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
        T sum = T(0);
        for(int n = 0; n < size; ++n){
            sum += x_ptr[n];
        }
        y_ptr[0] = sum / T(size);
    }
private:
    TensorMemRef *x;
    TensorMemRef *y;
    int size;
};

REGIST_OP_ALGO(ReduceMean)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CPU::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReduceMean<Device::CPU, type::float32>;
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
        T dy_val = dy_ptr[0];

        for(int n = 0; n < size; ++n){
            dx_ptr[n] = dy_val * scale;
        }
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
    .Device(Device::CPU::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReduceMeanGrad<Device::CPU, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace mlfe
