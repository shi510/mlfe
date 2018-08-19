#include "../core/op_algo.h"
#include "../core/device.h"
#include "../core/tensor_mem_ref.h"
#include "../math/blas.h"
#include "../math/transform.h"
#include "../device_context/cuda_context.h"

namespace mlfe{

template <class Dev, class Tp>
class MaxPool : public OpAlgo{
using T = typename Tp::T;
public:
    MaxPool(OpAlgoContext *oac) : OpAlgo(oac){
        using IntVec = std::vector<type::int32::T>;
        x = oac->GetVar("X");
        idx = oac->GetVar("IDX");
        y = oac->GetVar("Y");
        filters_hw = oac->GetAttr<IntVec>("filters_hw");
        strides = oac->GetAttr<IntVec>("strides");
        pads = oac->GetAttr<IntVec>("pads");

        in_c = x->Shape()[1];
        in_h = x->Shape()[2];
        in_w = x->Shape()[3];
        out_h = y->Shape()[2];
        out_w = y->Shape()[3];
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto idx_ptr = idx->Data<int>();
        auto y_ptr = y->Data<T>();

        math::MaxPool<T, CUDAContext>(
            y->Size(), x_ptr,
            in_c, in_h, in_w,
            out_h, out_w,
            filters_hw[0], filters_hw[1], strides[0], strides[1],
            pads[0], pads[1],
            y_ptr, idx_ptr
            );
    }
private:
    TensorMemRef *x;
    TensorMemRef *idx;
    TensorMemRef *y;
    int in_c, in_h, in_w;
    int out_h, out_w;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
};

REGIST_OP_ALGO(MaxPool)
    .Input("X", type::float32::string)
    .Output("IDX", type::float32::string)
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = MaxPool<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class MaxPoolGrad : public OpAlgo{
    using T = typename Tp::T;
public:
    MaxPoolGrad(OpAlgoContext *oac) : OpAlgo(oac){
        using IntVec = std::vector<type::int32::T>;
        x = oac->GetVar("X");
        idx = oac->GetVar("IDX");
        y = oac->GetVar("Y");
        dy = oac->GetVar("dY");
        dx = oac->GetVar("dX");
        filters_hw = oac->GetAttr<IntVec>("filters_hw");
        strides = oac->GetAttr<IntVec>("strides");
        pads = oac->GetAttr<IntVec>("pads");

        in_c = x->Shape()[1];
        in_h = x->Shape()[2];
        in_w = x->Shape()[3];
        out_h = y->Shape()[2];
        out_w = y->Shape()[3];
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto idx_ptr = idx->Data<int>();
        auto y_ptr = y->Data<T>();
        auto dy_ptr = dy->Data<T>();
        auto dx_ptr = dx->Data<T>();

        math::MaxPoolGradient<T, CUDAContext>(
            dy->Size(),
            dy_ptr, idx_ptr,
            in_c, in_h, in_w,
            out_h, out_w,
            filters_hw[0], filters_hw[1],
            strides[0], strides[1],
            pads[0], pads[1],
            dx_ptr
            );
    }

private:
    TensorMemRef *x;
    TensorMemRef *idx;
    TensorMemRef *y;
    TensorMemRef *dy;
    TensorMemRef *dx;
    Device col_buf;
    int in_c, in_h, in_w;
    int out_h, out_w;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
};

REGIST_OP_GRAD_ALGO(MaxPool)
    .Input("X", type::float32::string)
    .Input("IDX", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = MaxPoolGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace mlfe
