#include "../core/op_algo.h"
#include "../core/device.h"
#include "../core/tensor_mem_ref.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../math/transform.h"
#include "../device_context/cpu_context.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace mlfe{ namespace algorithm_cpu{

template <class Dev, class Tp>
class Convolution : public OpAlgo{
using T = typename Tp::T;
using IntVec = std::vector<type::int32::T>;
using T4R = Eigen::Tensor<T, 4, Eigen::RowMajor>;
using T_MAP = Eigen::TensorMap<T4R>;
using ArrI4 = Eigen::array<int, 4>;
public:
    Convolution(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->GetVar("X");
        w = oac->GetVar("W");
        y = oac->GetVar("Y");
        filters = oac->GetAttr<type::int32::T>("filters");
        filters_hw = oac->GetAttr<IntVec>("filters_hw");
        strides = oac->GetAttr<IntVec>("strides");
        pads = oac->GetAttr<IntVec>("pads");

        y_t = T4R(
            y->Shape()[0],
            y->Shape()[2],
            y->Shape()[3],
            y->Shape()[1]
        );
        contract_shape[0] = Eigen::IndexPair<int>(1, 0);
        pre_contract_shape[1] = filters_hw[0] * filters_hw[1] * x->Shape()[1];
        pre_contract_shape[0] = y->Size() / filters;
        kernel_shape[0] = filters_hw[0] * filters_hw[1] * x->Shape()[1];
        kernel_shape[1] = filters;
    }

    void Compute() override{
        T4R x_t = T_MAP(
            x->Data<T>(),
            x->Shape()[0],
            x->Shape()[1],
            x->Shape()[2],
            x->Shape()[3]
        ).shuffle(ArrI4{{ 0, 2, 3, 1 } });

        T4R w_t = T_MAP(
            w->Data<T>(),
            w->Shape()[0],
            w->Shape()[1],
            w->Shape()[2],
            w->Shape()[3]
        ).shuffle(ArrI4{{ 2, 3, 1, 0 } });

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
            y->Data<T>(),
            y->Shape()[0],
            y->Shape()[1],
            y->Shape()[2],
            y->Shape()[3]
        ) = y_t.shuffle(ArrI4{{0, 3, 1, 2 } });
    }
private:
    TensorMemRef *x;
    TensorMemRef *w;
    TensorMemRef *y;
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
    .Device(Device::CPU::string)
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Convolution<Device::CPU, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class ConvolutionGrad : public OpAlgo{
using T = typename Tp::T;
using IntVec = std::vector<type::int32::T>;
public:
    ConvolutionGrad(OpAlgoContext *oac) : OpAlgo(oac){
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

        math::set<T, CPUContext>(
            dx->Size(),
            T(0),
            dx_ptr
            );
        math::set<T, CPUContext>(
            dw->Size(),
            T(0),
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
                T(1), dy_ptr, n,
                col_ptr, n,
                T(1), dw_ptr, k, nullptr
                );

            /*
            * Calculate loss to propagate through bottom.
            * w({filters, kernel_size})^T * dy({filters, out_size})
            *  = col({kernel_size, out_size})
            */
            math::gemm<T, CPUContext>(
                true, false, k, n, m,
                T(1), w_ptr, k,
                dy_ptr, n,
                T(0), col_ptr, n, nullptr
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
};

REGIST_OP_GRAD_ALGO(Convolution)
    .Input("X", type::float32::string)
    .Input("W", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dW", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CPU::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ConvolutionGrad<Device::CPU, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
