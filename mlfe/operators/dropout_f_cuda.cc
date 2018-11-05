#include "../core/op_algo.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../device_context/cuda_context.h"
#include "../core/device.h"

namespace mlfe{ namespace algorithm_cuda{

template <class Tp>
class Dropout : public OpAlgo{
using T = typename Tp::T;
public:
    Dropout(OpAlgoContext *oac) : OpAlgo(oac){
        y = oac->get_output(0);
        x = y.get_children()[0];
        prob = y.get_children()[1];
        mask = y.get_children()[2];
        drop_ratio = prob.host_data<T>()[0];
        drop_ratio_inv = T(1) / (T(1) - drop_ratio);
        size = x.Size();
    }

    void Compute() override{
        auto x_ptr = x.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        auto mask_ptr = mask.mutable_device_data<T>();
        if(drop_ratio != 0){
            math::bernoulli_distribution<T, CUDAContext>(size, drop_ratio, mask_ptr);
            math::scal<T, CUDAContext>(size, drop_ratio_inv, mask_ptr, y_ptr);
            math::elementwise_mul<T, CUDAContext>(size, x_ptr, y_ptr, y_ptr);
        }
        else{
            copy(x.get_memory(), y.get_memory());
        }
    }
private:
    Tensor x;
    Tensor y;
    Tensor mask;
    Tensor prob;
    T drop_ratio, drop_ratio_inv;
    int size;
};

REGIST_OP_ALGO(Dropout)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Output("Mask", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Dropout<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class DropoutGrad : public OpAlgo{
using T = typename Tp::T;
public:
    DropoutGrad(OpAlgoContext *oac) : OpAlgo(oac){
        dx = oac->get_output(0);
        prob = dx.get_children()[0];
        mask = dx.get_children()[1];
        dy = dx.get_children()[2];
        drop_ratio = prob.host_data<T>()[0];
        drop_ratio_inv = T(1) / (T(1) - drop_ratio);
        size = dy.Size();
    }

    void Compute() override{
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();
        auto mask_ptr = mask.device_data<T>();
        math::scal<T, CUDAContext>(size, drop_ratio_inv, mask_ptr, dx_ptr);
        math::elementwise_mul<T, CUDAContext>(size, dy_ptr, dx_ptr, dx_ptr);
    }

private:
    Tensor prob;
    Tensor mask;
    Tensor dy;
    Tensor dx;
    T drop_ratio, drop_ratio_inv;
    int size;
};

REGIST_OP_GRAD_ALGO(Dropout)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("Mask", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = DropoutGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
