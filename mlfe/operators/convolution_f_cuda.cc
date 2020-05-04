#include "../core/op_algo.h"
#include "../core/device.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../math/transform.h"
#include "../device_context/cuda_context.h"

namespace mlfe{
namespace algorithm_cuda{

template <class Tp>
class Convolution : public OpAlgo{
using T = typename Tp::T;
public:
    Convolution(OpAlgoContext *oac) : OpAlgo(oac, "Convolution"){
        y = oac->get_output(0);
        x = y.get_children()[0];
        w = y.get_children()[1];
        auto x_shape = x.shape();
        auto w_shape = w.shape();
        auto y_shape = y.shape();
        filters_hw.resize(2);
        filters_hw[0] = w_shape[2];
        filters_hw[1] = w_shape[3];
        strides = oac->get_attr<std::vector<int>>("strides");
        pads = oac->get_attr<std::vector<int>>("pads");

        // Output Filters.
        m = w_shape[0];
        // Output Feature Map Size.
        n = y_shape[2] * y_shape[3];
        // Weight Size.
        k = w_shape[1] * w_shape[2] * w_shape[3];

        batch = x_shape[0];
        in_c = x_shape[1];
        in_h = x_shape[2];
        in_w = x_shape[3];

        col_buf = create_memory(k * n * Tp::size);
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = x.device_data<T>();
        auto w_ptr = w.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        auto col_buf_ptr = col_buf->mutable_device_data<T>();
        for(int i = 0; i < batch; ++i){
            /*
            * image to column in range on kernel size.
            */
            math::im2col<T, CUDAContext>(
                in_c, in_h, in_w,
                filters_hw[0], filters_hw[1],
                strides[0], pads[0],
                x_ptr, col_buf_ptr);

            /*
            * convolution with kernel.
            * kernel is learnable variable.
            * _w({filters, _kernel_size}) * x_col({_kernel_size, out_size})
            *  = _y({filters, out_size})
            */
            math::gemm<T, CUDAContext>(false, false,
                m, n, k,
                static_cast<T>(1), w_ptr, k,
                col_buf_ptr, n,
                static_cast<T>(0), y_ptr, n, &cuda);

            /*
            * next batch.
            */
            x_ptr += x.size() / batch;
            y_ptr += n * m;
        }
    }

private:
    Tensor y;
    Tensor x;
    Tensor w;
    memory_ptr col_buf;
    int m, n, k, batch;
    int in_c, in_h, in_w;
    type::int32::T filters;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
    CUDAContext cuda;
};

REGIST_OP_ALGO(Convolution)
    .Input("X", type::float32::string)
    .Input("W", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CUDA")
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

        //math::set<T, CUDAContext>(
        //    dx_t.size(),
        //    static_cast<T>(0),
        //    dx_ptr
        //    );

        for(int i = 0; i < batch; ++i){
            /*
            * Calculate loss to propagate through bottom.
            * w({filters, kernel_size})^T * dy({filters, out_size})
            *  = col({kernel_size, out_size})
            */
            math::gemm<T, CUDAContext>(
                true, false, k, n, m,
                static_cast<T>(1), w_ptr, k,
                dy_ptr, n,
                static_cast<T>(0), col_ptr, n, &cuda
                );

            math::col2im<T, CUDAContext>(
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
    CUDAContext cuda;
};

REGIST_OP_GRAD_ALGO(Conv2DGradientInput)
    .Input("W", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CUDA")
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

        math::set<T, CUDAContext>(
            dw.size(),
            static_cast<T>(0),
            dw_ptr
            );

        for(int i = 0; i < batch; ++i){
            math::im2col<T, CUDAContext>(
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
            math::gemm<T, CUDAContext>(
                false, true, m, k, n,
                static_cast<T>(1), dy_ptr, n,
                col_ptr, n,
                static_cast<T>(1), dw_ptr, k, &cuda
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
    CUDAContext cuda;
};

REGIST_OP_GRAD_ALGO(Conv2DGradientFilter)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dW", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Conv2DGradientFilter<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
