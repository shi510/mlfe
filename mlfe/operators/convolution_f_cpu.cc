#include "mlfe/core/op_algo.h"
#include "mlfe/core/device.h"
#include "mlfe/math/blas.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/math/transform.h"
#include "mlfe/device_context/cpu_context.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace mlfe{
namespace algorithm_cpu{

template <class Tp>
class Convolution : public OpAlgo{
using T = typename Tp::T;
using IntVec = std::vector<type::int32::T>;
using T4R = Eigen::Tensor<T, 4, Eigen::RowMajor>;
using CT4R = Eigen::Tensor<const T, 4, Eigen::RowMajor>;
using T_MAP = Eigen::TensorMap<T4R>;
using CT_MAP = Eigen::TensorMap<CT4R>;
using ArrI4 = Eigen::array<int, 4>;
public:
    Convolution(OpAlgoContext *oac) : OpAlgo(oac, "Convolution"){
        y = oac->get_output(0);
        x = y.get_children()[0];
        w = y.get_children()[1];
        filters = w.shape()[0];
        filters_hw.resize(2);
        filters_hw[0] = w.shape()[2];
        filters_hw[1] = w.shape()[3];
        strides = oac->get_attr<IntVec>("strides");
        pads = oac->get_attr<IntVec>("pads");

        y_t = T4R(
            y.shape()[0],
            y.shape()[2],
            y.shape()[3],
            y.shape()[1]
        );
        contract_shape[0] = Eigen::IndexPair<int>(1, 0);
        pre_contract_shape[1] = filters_hw[0] * filters_hw[1] * x.shape()[1];
        pre_contract_shape[0] = y.size() / filters;
        kernel_shape[0] = filters_hw[0] * filters_hw[1] * x.shape()[1];
        kernel_shape[1] = filters;
    }

    void Compute(op_algo_runtime_context& rc) override{
        T4R x_t = T_MAP(
            x.mutable_device_data<T>(),
            x.shape()[0],
            x.shape()[1],
            x.shape()[2],
            x.shape()[3]
        ).shuffle(ArrI4{{0, 2, 3, 1}});

        T4R w_t = T_MAP(
            w.mutable_device_data<T>(),
            w.shape()[0],
            w.shape()[1],
            w.shape()[2],
            w.shape()[3]
        ).shuffle(ArrI4{{2, 3, 1, 0}});

        y_t = x_t.extract_image_patches(
            filters_hw[0], filters_hw[1],
            strides[0], strides[1],
            1, 1, 1, 1,
            pads[0], pads[0],
            pads[1], pads[1], 0
        ).reshape(
            pre_contract_shape
        ).contract(
            w_t.reshape(kernel_shape),
            contract_shape
        ).reshape(y_t.dimensions());

        T_MAP(
            y.mutable_device_data<T>(),
            y.shape()[0],
            y.shape()[1],
            y.shape()[2],
            y.shape()[3]
        ) = y_t.shuffle(ArrI4{{0, 3, 1, 2}});
    }

private:
    Tensor x;
    Tensor w;
    Tensor y;
    T4R y_t;
    Eigen::array<Eigen::IndexPair<int>, 1> contract_shape;
    Eigen::array<int, 2> pre_contract_shape;
    Eigen::array<int, 2> kernel_shape;
    type::int32::T filters;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
};

REGIST_OP_ALGO(Convolution)
    .Input("X", type::float32::string)
    .Input("W", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Convolution<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();
    
template <class Tp>
class Conv2DGradientInput : public OpAlgo{
using T = typename Tp::T;
public:
    Conv2DGradientInput(OpAlgoContext *oac) : OpAlgo(oac){
        using IntVec = std::vector<type::int32::T>;
        dx = oac->get_output(0);
        w = dx.get_children()[0];
        dy = dx.get_children()[1];
        filters = w.shape()[0];
        filters_hw.resize(2);
        filters_hw[0] = w.shape()[2];
        filters_hw[1] = w.shape()[3];
        strides = oac->get_attr<IntVec>("strides");
        pads = oac->get_attr<IntVec>("pads");

        batch = dx.shape()[0];
        in_c = dx.shape()[1];
        in_h = dx.shape()[2];
        in_w = dx.shape()[3];

        // Output Filters.
        m = filters;
        // Output Feature Map Size.
        n = dy.shape()[2] * dy.shape()[3];
        // Weight Size.
        k = w.shape()[1] * filters_hw[1] * filters_hw[0];

        col_buf = create_memory(k * n * Tp::size);
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto w_ptr = w.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();
        auto col_ptr = col_buf->mutable_device_data<T>();

        math::set<T, CPUContext>(
            dx.size(),
            static_cast<T>(0),
            dx_ptr
            );

        for(int i = 0; i < batch; ++i){
            /*
            * Calculate loss to propagate through bottom.
            * w({filters, kernel_size})^T * dy({filters, out_size})
            *  = col({kernel_size, out_size})
            */
            math::gemm<T, CPUContext>(
                true, false, k, n, m,
                static_cast<T>(1), w_ptr, k,
                dy_ptr, n,
                static_cast<T>(0), col_ptr, n, nullptr
                );

            math::col2im<T, CPUContext>(
                col_ptr,
                in_c, in_h, in_w,
                filters_hw[0], strides[0], pads[0],
                dx_ptr
                );

            /*
            * next batch.
            */
            dx_ptr += dx.size() / batch;
            dy_ptr += n * m;
        }
    }

private:
    Tensor w;
    Tensor dx;
    Tensor dy;
    memory_ptr col_buf;
    int m, n, k, batch;
    int in_c, in_h, in_w;
    type::int32::T filters;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
};

REGIST_OP_GRAD_ALGO(Conv2DGradientInput)
    .Input("W", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Conv2DGradientInput<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class Conv2DGradientFilter : public OpAlgo{
using T = typename Tp::T;
public:
    Conv2DGradientFilter(OpAlgoContext *oac) : OpAlgo(oac){
        using IntVec = std::vector<type::int32::T>;
        dw = oac->get_output(0);
        x = dw.get_children()[0];
        dy = dw.get_children()[1];
        filters = dw.shape()[0];
        filters_hw.resize(2);
        filters_hw[0] = dw.shape()[2];
        filters_hw[1] = dw.shape()[3];
        strides = oac->get_attr<IntVec>("strides");
        pads = oac->get_attr<IntVec>("pads");

        batch = x.shape()[0];
        in_c = x.shape()[1];
        in_h = x.shape()[2];
        in_w = x.shape()[3];

        // Output Filters.
        m = filters;
        // Output Feature Map Size.
        n = dy.shape()[2] * dy.shape()[3];
        // Weight Size.
        k = x.shape()[1] * filters_hw[1] * filters_hw[0];

        col_buf = create_memory(k * n * Tp::size);
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = x.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dw_ptr = dw.mutable_device_data<T>();
        auto col_ptr = col_buf->mutable_device_data<T>();

        math::set<T, CPUContext>(
            dw.size(),
            static_cast<T>(0),
            dw_ptr
            );

        for(int i = 0; i < batch; ++i){
            math::im2col<T, CPUContext>(
                in_c, in_h, in_w,
                filters_hw[0], filters_hw[1],
                strides[0], pads[0],
                x_ptr, col_ptr
                );

            /*
            * Calculate gradients of weights.
            * kernel_size ={kernel_h, kernel_w, channel_of_x} = k
            * filters ={number of feature map channel} = m
            * out_size ={y_h, y_w} = n
            * dy({filters, out_size}) * col({kernel_size, out_size})^T
            *  = dw({filters, kernel_size})
            */
            math::gemm<T, CPUContext>(
                false, true, m, k, n,
                static_cast<T>(1), dy_ptr, n,
                col_ptr, n,
                static_cast<T>(1), dw_ptr, k, nullptr
                );

            /*
            * next batch.
            */
            x_ptr += x.size() / batch;
            dy_ptr += n * m;
        }
    }

private:
    Tensor x;
    Tensor dy;
    Tensor dw;
    memory_ptr col_buf;
    int m, n, k, batch;
    int in_c, in_h, in_w;
    type::int32::T filters;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
};

REGIST_OP_GRAD_ALGO(Conv2DGradientFilter)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dW", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Conv2DGradientFilter<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
