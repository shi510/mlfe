#include "../core/op_algo.h"
#include "../core/device.h"
#include "../math/blas.h"
#include "../math/transform.h"
#include "../device_context/cuda_context.h"

namespace mlfe{
namespace algorithm_cuda{

template <class Tp>
class MaxPool : public OpAlgo{
using T = typename Tp::T;
public:
    MaxPool(OpAlgoContext *oac) : OpAlgo(oac, "MaxPool"){
        using IntVec = std::vector<type::int32::T>;
        y = oac->get_output(0);
        x = y.get_children()[0];
        filters_hw = oac->get_attr<IntVec>("kernel");
        strides = oac->get_attr<IntVec>("stride");
        pads = oac->get_attr<IntVec>("padding");
        idx = oac->get_attr<Tensor>("idx");

        in_c = x.shape()[1];
        in_h = x.shape()[2];
        in_w = x.shape()[3];
        out_h = y.shape()[2];
        out_w = y.shape()[3];
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = x.device_data<T>();
        auto idx_ptr = idx.mutable_device_data<int>();
        auto y_ptr = y.mutable_device_data<T>();

        math::MaxPool<T, CUDAContext>(
            y.size(), x_ptr,
            in_c, in_h, in_w,
            out_h, out_w,
            filters_hw[0], filters_hw[1], strides[0], strides[1],
            pads[0], pads[1],
            y_ptr, idx_ptr
            );
    }
private:
    Tensor x;
    Tensor idx;
    Tensor y;
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
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = MaxPool<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class MaxPoolGrad : public OpAlgo{
    using T = typename Tp::T;
public:
    MaxPoolGrad(OpAlgoContext *oac) : OpAlgo(oac){
        using IntVec = std::vector<type::int32::T>;
        dx = oac->get_output(0);
        x = dx.get_children()[0];
        dy = dx.get_children()[2];
        filters_hw = oac->get_attr<IntVec>("kernel");
        strides = oac->get_attr<IntVec>("stride");
        pads = oac->get_attr<IntVec>("padding");
        idx = oac->get_attr<Tensor>("idx");

        in_c = x.shape()[1];
        in_h = x.shape()[2];
        in_w = x.shape()[3];
        out_h = dy.shape()[2];
        out_w = dy.shape()[3];
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = x.device_data<T>();
        auto idx_ptr = idx.device_data<int>();
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();

        math::MaxPoolGradient<T, CUDAContext>(
            dy.size(),
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
    Tensor x;
    Tensor idx;
    Tensor y;
    Tensor dy;
    Tensor dx;
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
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = MaxPoolGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
