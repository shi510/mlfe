#include "../core/op_algo.h"
#include "../core/device.h"
#include "../core/tensor_mem_ref.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../math/transform.h"
#include "../device_context/cuda_context.h"
#include <iostream>

namespace mlfe{ namespace algorithm_cuda{

template <class Dev, class Tp>
class Convolution : public OpAlgo{
using T = typename Tp::T;
public:
    Convolution(OpAlgoContext *oac) : OpAlgo(oac){
        using IntVec = std::vector<type::int32::T>;
        x = oac->GetVar("X");
        w = oac->GetVar("W");
        y = oac->GetVar("Y");
        filters = oac->GetAttr<type::int32::T>("filters");
        filters_hw = oac->GetAttr<IntVec>("filters_hw");
        strides = oac->GetAttr<IntVec>("strides");
        pads = oac->GetAttr<IntVec>("pads");

        batch = x->Shape()[0];
        // Output Filters.
        m = filters;
        // Output Feature Map Size.
        n = y->Shape()[2] * y->Shape()[3];
        // Weight Size.
        k = x->Shape()[1] * filters_hw[1] * filters_hw[0];

        in_c = x->Shape()[1];
        in_h = x->Shape()[2];
        in_w = x->Shape()[3];

        col_buf = oac->GetDevice().CreateDeviceMemory();
        col_buf.Allocate(k * n * Tp::size);
    }

    void Compute() override{
        const T *x_ptr = x->Data<T>();
        const T *w_ptr = w->Data<T>();
        T *y_ptr = y->Data<T>();
        T *col_ptr = col_buf.Data<T>();


        for(int i = 0; i < batch; ++i){
            /*
            * image to column in range on kernel size.
            */
            math::im2col<T, CUDAContext>(
                in_c, in_h, in_w,
                filters_hw[0], filters_hw[1],
                strides[0], pads[0],
                x_ptr, col_ptr);

            /*
            * convolution with kernel.
            * kernel is learnable variable.
            * _w({filters, _kernel_size}) * x_col({_kernel_size, out_size})
            *  = _y({filters, out_size})
            */
            math::gemm<T, CUDAContext>(false, false,
                m, n, k,
                static_cast<T>(1), w_ptr, k,
                col_ptr, n,
                static_cast<T>(0), y_ptr, n, &cuda);

            /*
            * next batch.
            */
            x_ptr += x->Size() / batch;
            y_ptr += n * m;
        }
    }
private:
    TensorMemRef *x;
    TensorMemRef *w;
    TensorMemRef *y;
    DeviceMemory col_buf;
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
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Convolution<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class ConvolutionGrad : public OpAlgo{
using T = typename Tp::T;
public:
    ConvolutionGrad(OpAlgoContext *oac) : OpAlgo(oac){
        using IntVec = std::vector<type::int32::T>;
        x = oac->GetVar("X");
        w = oac->GetVar("W");
        y = oac->GetVar("Y");
        dy = oac->GetVar("dY");
        dw = oac->GetVar("dW");
        dx = oac->GetVar("dX");
        filters = oac->GetAttr<type::int32::T>("filters");
        filters_hw = oac->GetAttr<IntVec>("filters_hw");
        strides = oac->GetAttr<IntVec>("strides");
        pads = oac->GetAttr<IntVec>("pads");

        batch = x->Shape()[0];
        in_c = x->Shape()[1];
        in_h = x->Shape()[2];
        in_w = x->Shape()[3];

        // Output Filters.
        m = filters;
        // Output Feature Map Size.
        n = y->Shape()[2] * y->Shape()[3];
        // Weight Size.
        k = x->Shape()[1] * filters_hw[1] * filters_hw[0];

        col_buf = oac->GetDevice().CreateDeviceMemory();
        col_buf.Allocate(k * n * Tp::size);
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto w_ptr = w->Data<T>();
        auto y_ptr = y->Data<T>();
        auto dy_ptr = dy->Data<T>();
        auto dw_ptr = dw->Data<T>();
        auto dx_ptr = dx->Data<T>();
        T *col_ptr = col_buf.Data<T>();

        math::set<T, CUDAContext>(
            dx->Size(),
            static_cast<T>(0),
            dx_ptr
            );
        math::set<T, CUDAContext>(
            dw->Size(),
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
            x_ptr += x->Size() / batch;
            dx_ptr += dx->Size() / batch;
            dy_ptr += n * m;
        }
    }

private:
    TensorMemRef *x;
    TensorMemRef *w;
    TensorMemRef *y;
    TensorMemRef *dy;
    TensorMemRef *dw;
    TensorMemRef *dx;
    DeviceMemory col_buf;
    int m, n, k, batch;
    int in_c, in_h, in_w;
    type::int32::T filters;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
    CUDAContext cuda;
};

REGIST_OP_GRAD_ALGO(Convolution)
    .Input("X", type::float32::string)
    .Input("W", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dW", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ConvolutionGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
