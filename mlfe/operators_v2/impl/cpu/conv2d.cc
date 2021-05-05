#include "mlfe/operators_v2/conv2d.h"
#include "mlfe/core/op_kernel.h"
#include "mlfe/device_context/cpu_context.h"
#include "mlfe/core/device.h"
#include "mlfe/math/blas.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/math/transform.h"
#include "mlfe/device_context/cpu_context.h"
#include "mlfe/operators/convolution_utils.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace mlfe{
namespace operators_v2{
namespace {

template <typename T>
struct conv2d_nchw
{
    using IntVec = std::vector<type::int32::T>;
    using T4R = Eigen::Tensor<T, 4, Eigen::RowMajor>;
    using CT4R = Eigen::Tensor<const T, 4, Eigen::RowMajor>;
    using T_MAP = Eigen::TensorMap<T4R>;
    using CT_MAP = Eigen::TensorMap<CT4R>;
    using ArrI4 = Eigen::array<int, 4>;

    static void run(Tensor x,
        Tensor kernel,
        Tensor y,
        std::vector<int32_t> strides,
        std::vector<int32_t> paddings
        )
    {
        T4R y_t;
        Eigen::array<Eigen::IndexPair<int>, 1> contract_shape;
        Eigen::array<int, 2> pre_contract_shape;
        Eigen::array<int, 2> kernel_shape;
        type::int32::T out_channels = kernel.shape()[0];
        std::vector<type::int32::T> kernel_hw =
            { kernel.shape()[2], kernel.shape()[3]};
        y_t = T4R(y.shape()[0], y.shape()[2], y.shape()[3], y.shape()[1]);
        contract_shape[0] = Eigen::IndexPair<int>(1, 0);
        pre_contract_shape[1] = kernel_hw[0] * kernel_hw[1] * x.shape()[1];
        pre_contract_shape[0] = y.size() / out_channels;
        kernel_shape[0] = kernel_hw[0] * kernel_hw[1] * x.shape()[1];
        kernel_shape[1] = out_channels;

        T4R x_t = T_MAP(
            x.mutable_device_data<T>(),
            x.shape()[0],
            x.shape()[1],
            x.shape()[2],
            x.shape()[3]
        ).shuffle(ArrI4{{0, 2, 3, 1}});

        T4R w_t = T_MAP(
            kernel.mutable_device_data<T>(),
            kernel.shape()[0],
            kernel.shape()[1],
            kernel.shape()[2],
            kernel.shape()[3]
        ).shuffle(ArrI4{{2, 3, 1, 0}});

        y_t = x_t.extract_image_patches(
            kernel_hw[0], kernel_hw[1],
            strides[0], strides[1],
            1, 1, 1, 1,
            paddings[0], paddings[0],
            paddings[1], paddings[1], 0
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

};
    
template <class T>
struct conv2d_nchw_input_grad
{
    static void run(
        Tensor kernel,
        Tensor dy,
        Tensor dx,
        std::vector<int32_t> strides,
        std::vector<int32_t> paddings
        )
    {
        using IntVec = std::vector<type::int32::T>;
        memory_ptr col_buf;
        int m, n, k, batch;
        int in_c, in_h, in_w;
        std::vector<type::int32::T> filters_hw;
        filters_hw.resize(2);
        filters_hw[0] = kernel.shape()[2];
        filters_hw[1] = kernel.shape()[3];

        batch = dx.shape()[0];
        in_c = dx.shape()[1];
        in_h = dx.shape()[2];
        in_w = dx.shape()[3];

        // output channels.
        m = kernel.shape()[0];
        // output height * output width
        n = dy.shape()[2] * dy.shape()[3];
        // input channels * kernel height * kernel width
        k = kernel.shape()[1] * filters_hw[1] * filters_hw[0];

        col_buf = create_memory(k * n * sizeof(T));

        auto w_ptr = kernel.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();
        auto col_ptr = col_buf->mutable_device_data<T>();

        math::set<T, CPUContext>(dx.size(), static_cast<T>(0), dx_ptr);

        for(int i = 0; i < batch; ++i){
            /*
            * Calculate loss to propagate through bottom.
            * w({out_channels, kernel_size})^T * dy({out_channel, out_size})
            *  = col({kernel_size, out_size})
            */
            math::gemm<T, CPUContext>(
                true, false, k, n, m,
                static_cast<T>(1), w_ptr, k,
                dy_ptr, n,
                static_cast<T>(0), col_ptr, n, nullptr
                );

            math::col2im_nchw<T, CPUContext>(
                col_ptr,
                in_c, in_h, in_w,
                filters_hw[0], strides[0], paddings[0],
                dx_ptr
                );

            /*
            * next batch.
            */
            dx_ptr += dx.size() / batch;
            dy_ptr += n * m;
        }
    }

};

template <class T>
struct conv2d_nchw_kernel_grad{

    static void run(
        Tensor x,
        Tensor dy,
        Tensor dkernel,
        std::vector<int32_t> strides,
        std::vector<int32_t> paddings
        )
    {
        using IntVec = std::vector<type::int32::T>;
        memory_ptr col_buf;
        int m, n, k, batch;
        int in_c, in_h, in_w;
        std::vector<type::int32::T> kernel_hw =
            {dkernel.shape()[2], dkernel.shape()[3]};
        batch = x.shape()[0];
        in_c = x.shape()[1];
        in_h = x.shape()[2];
        in_w = x.shape()[3];

        // output channels.
        m = dkernel.shape()[0];
        // output height * width
        n = dy.shape()[2] * dy.shape()[3];
        // in_channels * kernel_height * kernel_width
        k = x.shape()[1] * kernel_hw[1] * kernel_hw[0];

        col_buf = create_memory(k * n * sizeof(T));

        auto x_ptr = x.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dw_ptr = dkernel.mutable_device_data<T>();
        auto col_ptr = col_buf->mutable_device_data<T>();

        math::set<T, CPUContext>(dkernel.size(), static_cast<T>(0), dw_ptr);

        for(int i = 0; i < batch; ++i){
            math::im2col_nchw<T, CPUContext>(
                in_c, in_h, in_w,
                kernel_hw[0], kernel_hw[1],
                strides[0], paddings[0],
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
};

template <typename T>
struct conv2d_nhwc_op
{
    using IntVec = std::vector<type::int32::T>;
    using T4R = Eigen::Tensor<T, 4, Eigen::RowMajor>;
    using CT4R = Eigen::Tensor<const T, 4, Eigen::RowMajor>;
    using T_MAP = Eigen::TensorMap<T4R>;
    using CT_MAP = Eigen::TensorMap<CT4R>;
    using ArrI4 = Eigen::array<int, 4>;
    typedef typename Eigen::internal::traits<
        Eigen::Tensor<T, 4, Eigen::RowMajor>>::Index TensorIndex;

    static void run(Tensor x,
        Tensor kernel,
        Tensor y,
        std::vector<int32_t> strides,
        std::vector<int32_t> paddings
        )
    {
        T_MAP y_t = T_MAP(
            y.mutable_device_data<T>(),
            y.shape()[0], y.shape()[1], y.shape()[2], y.shape()[3]);
        Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_shape;
        Eigen::array<TensorIndex, 2> pre_contract_shape;
        Eigen::array<TensorIndex, 2> kernel_shape;
        int32_t out_channels = kernel.shape()[3];
        std::vector<int> kernel_hw =
            { kernel.shape()[0], kernel.shape()[1]};
        contract_shape[0] = Eigen::IndexPair<TensorIndex>(1, 0);
        pre_contract_shape[1] = kernel_hw[0] * kernel_hw[1] * x.shape()[3];
        pre_contract_shape[0] = y.size() / out_channels;
        kernel_shape[0] = kernel_hw[0] * kernel_hw[1] * x.shape()[3];
        kernel_shape[1] = out_channels;

        CT_MAP x_t = CT_MAP(
            x.device_data<T>(),
            x.shape()[0],
            x.shape()[1],
            x.shape()[2],
            x.shape()[3]
        );

        CT_MAP w_t = CT_MAP(
            kernel.device_data<T>(),
            kernel.shape()[0],
            kernel.shape()[1],
            kernel.shape()[2],
            kernel.shape()[3]
        );

        y_t = x_t
            .extract_image_patches(
                kernel_hw[0], kernel_hw[1],
                strides[0], strides[1],
                1, 1, 1, 1,
                paddings[0], paddings[0],
                paddings[1], paddings[1], 0)
            .reshape(pre_contract_shape)
            .contract(w_t.reshape(kernel_shape), contract_shape)
            .reshape(y_t.dimensions());
    }
};

template <typename T>
struct conv2d_nhwc_input_grad_op
{
    static void run(
        Tensor kernel,
        Tensor dy,
        Tensor dx,
        std::vector<int32_t> strides,
        std::vector<int32_t> paddings
        )
    {
        using IntVec = std::vector<type::int32::T>;
        memory_ptr col_buf;
        int m, n, k, batch;
        int in_c, in_h, in_w;
        std::vector<type::int32::T> filters_hw;
        filters_hw.resize(2);
        filters_hw[0] = kernel.shape()[0];
        filters_hw[1] = kernel.shape()[1];

        batch = dx.shape()[0];
        in_h = dx.shape()[1];
        in_w = dx.shape()[2];
        in_c = dx.shape()[3];

        // output channels.
        m = kernel.shape()[3];
        // output height * output width
        n = dy.shape()[1] * dy.shape()[2];
        // input channels * kernel height * kernel width
        k = kernel.shape()[2] * filters_hw[0] * filters_hw[1];

        col_buf = create_memory(k * n * sizeof(T));

        auto w_ptr = kernel.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();
        auto col_ptr = col_buf->mutable_device_data<T>();

        math::set<T, CPUContext>(dx.size(), static_cast<T>(0), dx_ptr);

        for(int i = 0; i < batch; ++i){
            /*
            * Calculate loss to propagate through bottom.
            * w({kernel_size, out_channel}) * dy({out_size, out_channel})^T
            *  = col({kernel_size, out_size})
            */
            math::gemm<T, CPUContext>(
                false, true, k, n, m,
                static_cast<T>(1), w_ptr, k,
                dy_ptr, n,
                static_cast<T>(0), col_ptr, n, nullptr
                );

            math::col2im_nhwc<T, CPUContext>(
                col_ptr,
                in_c, in_h, in_w,
                filters_hw[0], strides[0], paddings[0],
                dx_ptr
                );

            /*
            * next batch.
            */
            dx_ptr += dx.size() / batch;
            dy_ptr += n * m;
        }
    }

};

template <typename T>
struct conv2d_nhwc_kernel_grad_op{

    static void run(
        Tensor x,
        Tensor dy,
        Tensor dkernel,
        std::vector<int32_t> strides,
        std::vector<int32_t> paddings
        )
    {
        using IntVec = std::vector<type::int32::T>;
        memory_ptr col_buf;
        int m, n, k, batch;
        int in_c, in_h, in_w;
        std::vector<type::int32::T> kernel_hw =
            {dkernel.shape()[0], dkernel.shape()[1]};
        batch = x.shape()[0];
        in_h = x.shape()[1];
        in_w = x.shape()[2];
        in_c = x.shape()[3];

        // output channels.
        m = dkernel.shape()[3];
        // output height * width
        n = dy.shape()[1] * dy.shape()[2];
        // in_channels * kernel_height * kernel_width
        k = x.shape()[3] * kernel_hw[0] * kernel_hw[1];

        col_buf = create_memory(k * n * sizeof(T));

        auto x_ptr = x.device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto dw_ptr = dkernel.mutable_device_data<T>();
        auto col_ptr = col_buf->mutable_device_data<T>();

        math::set<T, CPUContext>(dkernel.size(), static_cast<T>(0), dw_ptr);
        math::set<T, CPUContext>(k * n, static_cast<T>(0), col_ptr);

        for(int i = 0; i < batch; ++i){
            math::im2col_nhwc<T, CPUContext>(
                in_c, in_h, in_w,
                kernel_hw[0], kernel_hw[1],
                strides[0], paddings[0],
                x_ptr, col_ptr
                );

            /*
            * Calculate gradients of weights.
            * kernel_size ={kernel_h, kernel_w, channel_of_x} = k
            * filters ={number of feature map channel} = m
            * out_size ={y_h, y_w} = n
            * col({kernel_size, out_size}) * dy({filters, out_size})^T
            *  = dw({filters, kernel_size})
            */
            math::gemm<T, CPUContext>(
                false, true, k, m, n,
                static_cast<T>(1), col_ptr, n,
                dy_ptr, n,
                static_cast<T>(1), dw_ptr, k, nullptr
                );

            /*
            * next batch.
            */
            x_ptr += x.size() / batch;
            dy_ptr += n * m;
        }
    }
};

} // namespace anonymous

REGIST_OP_KERNEL(
    conv2d_fwd,
    conv2d_fwd_fn_t,
    conv2d_nhwc_op<float>::run
    );

REGIST_OP_KERNEL(
    conv2d_input_bwd,
    conv2d_input_bwd_fn_t,
    conv2d_nhwc_input_grad_op<float>::run
    );

REGIST_OP_KERNEL(
    conv2d_kernel_bwd,
    conv2d_kernel_bwd_fn_t,
    conv2d_nhwc_kernel_grad_op<float>::run
    );

} // namespace operators
} // namespace mlfe
