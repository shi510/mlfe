#include "../core/op_algo.h"
#include "../core/device.h"
#include "../math/blas.h"
#include "../math/transform.h"
#include "../device_context/cuda_context.h"
#include <cudnn.h>
#include <string>

namespace mlfe{
namespace algorithm_cudnn{

template <class Tp>
class Convolution : public OpAlgo{
using T = typename Tp::T;
public:
    void inline ASSERT_SUCCESS(cudnnStatus_t t){
        if(t != CUDNN_STATUS_SUCCESS){
            std::string e_msg;
            e_msg = "NOT CUDNN_STATUS_SUCCESS : ";
            e_msg += cudnnGetErrorString(t);
            throw e_msg;
        }
    }

    Convolution(OpAlgoContext *oac) : OpAlgo(oac, "Convolution"){
        using IntVec = std::vector<type::int32::T>;
        auto cudnn_type = [](std::string type_str){
            return type_str == type::float32::string ?
                CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
        };
        y = oac->get_output(0);
        x = y.get_children()[0];
        w = y.get_children()[1];
        strides = oac->get_attr<std::vector<int>>("strides");
        pads = oac->get_attr<std::vector<int>>("pads");

        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_x_desc);
        cudnnCreateFilterDescriptor(&_w_desc);
        cudnnCreateTensorDescriptor(&_y_desc);
        cudnnCreateConvolutionDescriptor(&_conv_desc);

        cudnnSetTensor4dDescriptor(
            _x_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]);

        cudnnSetFilter4dDescriptor(
            _w_desc,
            cudnn_type(Tp::string),
            CUDNN_TENSOR_NCHW,
            w.Shape()[0], w.Shape()[1], w.Shape()[2], w.Shape()[3]);

        cudnnSetTensor4dDescriptor(
            _y_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            y.Shape()[0], y.Shape()[1], y.Shape()[2], y.Shape()[3]);

        ASSERT_SUCCESS(cudnnSetConvolution2dDescriptor(
            _conv_desc,
            pads[0], pads[1],
            strides[0], strides[1], 1, 1,
            CUDNN_CROSS_CORRELATION,
            cudnn_type(Tp::string)
        ));

        ASSERT_SUCCESS(cudnnGetConvolutionForwardAlgorithm(
            _handle,
            _x_desc,
            _w_desc,
            _conv_desc,
            _y_desc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &_conv_algo
        ));

        ASSERT_SUCCESS(cudnnGetConvolutionForwardWorkspaceSize(
            _handle,
            _x_desc,
            _w_desc,
            _conv_desc,
            _y_desc,
            _conv_algo,
            &_ws_fwd_size
        ));

        cudaMalloc(&_ws_fwd, _ws_fwd_size);
    }

    void Compute() override{
        const float alpha = 1, beta = 0;
        ASSERT_SUCCESS(cudnnConvolutionForward(
            _handle,
            &alpha,
            _x_desc, x.device_data<void>(),
            _w_desc, w.device_data<void>(),
            _conv_desc, _conv_algo,
            _ws_fwd, _ws_fwd_size,
            &beta,
            _y_desc, y.mutable_device_data<void>()
        ));
    }

    ~Convolution(){
        cudnnDestroyTensorDescriptor(_x_desc);
        cudnnDestroyFilterDescriptor(_w_desc);
        cudnnDestroyTensorDescriptor(_y_desc);
        cudnnDestroyConvolutionDescriptor(_conv_desc);
        cudaFree(_ws_fwd);
        cudnnDestroy(_handle);
    }

private:
    Tensor x;
    Tensor w;
    Tensor y;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
    void *_ws_fwd;
    size_t _ws_fwd_size;
    cudnnHandle_t _handle;
    cudnnTensorDescriptor_t _x_desc;
    cudnnFilterDescriptor_t _w_desc;
    cudnnTensorDescriptor_t _y_desc;
    cudnnConvolutionFwdAlgo_t _conv_algo;
    cudnnConvolutionDescriptor_t _conv_desc;
};

REGIST_OP_ALGO(Convolution)
    .Input("X", type::float32::string)
    .Input("W", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CUDA(CUDNN)")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Convolution<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class Conv2DGradientInput : public OpAlgo{
using T = typename Tp::T;
    void inline ASSERT_SUCCESS(cudnnStatus_t t){
        if(t != CUDNN_STATUS_SUCCESS){
            std::string e_msg;
            e_msg = "NOT CUDNN_STATUS_SUCCESS : ";
            e_msg += cudnnGetErrorString(t);
            throw e_msg;
        }
    }
public:
    Conv2DGradientInput(OpAlgoContext *oac) : OpAlgo(oac, "Conv2DGradientInput"){
        using IntVec = std::vector<type::int32::T>;
        auto cudnn_type = [](std::string type_str){
            return type_str == type::float32::string ?
                CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
        };
        dx = oac->get_output(0);
        w = dx.get_children()[0];
        dy = dx.get_children()[1];
        strides = oac->get_attr<IntVec>("strides");
        pads = oac->get_attr<IntVec>("pads");

        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_dx_desc);
        cudnnCreateFilterDescriptor(&_w_desc);
        cudnnCreateTensorDescriptor(&_dy_desc);
        cudnnCreateConvolutionDescriptor(&_conv_desc);

        cudnnSetFilter4dDescriptor(_w_desc,
            cudnn_type(Tp::string),
            CUDNN_TENSOR_NCHW,
            w.Shape()[0], w.Shape()[1], w.Shape()[2], w.Shape()[3]);

        cudnnSetTensor4dDescriptor(_dy_desc, CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            dy.Shape()[0], dy.Shape()[1], dy.Shape()[2], dy.Shape()[3]);

        cudnnSetTensor4dDescriptor(_dx_desc, CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            dx.Shape()[0], dx.Shape()[1], dx.Shape()[2], dx.Shape()[3]);

        cudnnSetConvolution2dDescriptor(
            _conv_desc,
            pads[0], pads[1],
            strides[0], strides[1], 1, 1,
            CUDNN_CROSS_CORRELATION,
            cudnn_type(Tp::string)
        );

        cudnnGetConvolutionBackwardDataAlgorithm(
            _handle,
            _w_desc,
            _dy_desc,
            _conv_desc,
            _dx_desc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &_data_algo
        );

        cudnnGetConvolutionBackwardDataWorkspaceSize(
            _handle,
            _w_desc,
            _dy_desc,
            _conv_desc,
            _dx_desc,
            _data_algo,
            &_ws_data_size
        );

        cudaMalloc(&_ws_data, _ws_data_size);
    }

    void Compute() override{
        const float alpha = 1, beta = 0;

        ASSERT_SUCCESS(cudnnConvolutionBackwardData(
            _handle,
            &alpha,
            _w_desc, w.device_data<void>(),
            _dy_desc, dy.device_data<void>(),
            _conv_desc, _data_algo,
            _ws_data, _ws_data_size,
            &beta,
            _dx_desc, dx.mutable_device_data<void>()
        ));
    }

    ~Conv2DGradientInput(){
        cudnnDestroyTensorDescriptor(_dx_desc);
        cudnnDestroyFilterDescriptor(_w_desc);
        cudnnDestroyTensorDescriptor(_dy_desc);
        cudnnDestroyConvolutionDescriptor(_conv_desc);
        cudaFree(_ws_data);
        cudnnDestroy(_handle);
    }

private:
    Tensor w;
    Tensor dy;
    Tensor dx;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
    void *_ws_data;
    size_t _ws_data_size;
    cudnnHandle_t _handle;
    cudnnConvolutionFwdAlgo_t _conv_algo;
    cudnnConvolutionBwdDataAlgo_t _data_algo;
    cudnnTensorDescriptor_t _dx_desc;
    cudnnFilterDescriptor_t _w_desc;
    cudnnTensorDescriptor_t _dy_desc;
    cudnnConvolutionDescriptor_t _conv_desc;
};

REGIST_OP_GRAD_ALGO(Conv2DGradientInput)
    .Input("W", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CUDA(CUDNN)")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Conv2DGradientInput<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class Conv2DGradientFilter : public OpAlgo{
using T = typename Tp::T;
    void inline ASSERT_SUCCESS(cudnnStatus_t t){
        if(t != CUDNN_STATUS_SUCCESS){
            std::string e_msg;
            e_msg = "NOT CUDNN_STATUS_SUCCESS : ";
            e_msg += cudnnGetErrorString(t);
            throw e_msg;
        }
    }
public:
    Conv2DGradientFilter(OpAlgoContext *oac) : OpAlgo(oac, "Conv2DGradientFilter"){
            using IntVec = std::vector<type::int32::T>;
            auto cudnn_type = [](std::string type_str){
                return type_str == type::float32::string ?
                    CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
            };
            dw = oac->get_output(0);
            x = dw.get_children()[0];
            dy = dw.get_children()[1];
            strides = oac->get_attr<IntVec>("strides");
            pads = oac->get_attr<IntVec>("pads");

            cudnnCreate(&_handle);
            cudnnCreateTensorDescriptor(&_x_desc);
            cudnnCreateFilterDescriptor(&_dw_desc);
            cudnnCreateTensorDescriptor(&_dy_desc);
            cudnnCreateConvolutionDescriptor(&_conv_desc);

            cudnnSetTensor4dDescriptor(_x_desc, CUDNN_TENSOR_NCHW,
                cudnn_type(Tp::string),
                x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]);

            cudnnSetFilter4dDescriptor(_dw_desc,
                cudnn_type(Tp::string),
                CUDNN_TENSOR_NCHW,
                dw.Shape()[0], dw.Shape()[1], dw.Shape()[2], dw.Shape()[3]);

            cudnnSetTensor4dDescriptor(_dy_desc, CUDNN_TENSOR_NCHW,
                cudnn_type(Tp::string),
                dy.Shape()[0], dy.Shape()[1], dy.Shape()[2], dy.Shape()[3]);

            cudnnSetConvolution2dDescriptor(
                _conv_desc,
                pads[0], pads[1],
                strides[0], strides[1], 1, 1,
                CUDNN_CROSS_CORRELATION,
                cudnn_type(Tp::string)
            );

            cudnnGetConvolutionForwardAlgorithm(
                _handle,
                _x_desc,
                _dw_desc,
                _conv_desc,
                _dy_desc,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                0,
                &_conv_algo
            );

            cudnnGetConvolutionBackwardFilterAlgorithm(
                _handle,
                _x_desc,
                _dy_desc,
                _conv_desc,
                _dw_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                0,
                &_filter_algo
            );

            cudnnGetConvolutionBackwardFilterWorkspaceSize(
                _handle,
                _x_desc,
                _dy_desc,
                _conv_desc,
                _dw_desc,
                _filter_algo,
                &_ws_filter_size
            );

            cudaMalloc(&_ws_filter, _ws_filter_size);
    }

    void Compute() override{
        const float alpha = 1, beta = 0;

        ASSERT_SUCCESS(cudnnConvolutionBackwardFilter(
            _handle,
            &alpha, _x_desc, x.device_data<void>(),
            _dy_desc, dy.device_data<void>(),
            _conv_desc, _filter_algo,
            _ws_filter, _ws_filter_size,
            &beta,
            _dw_desc, dw.mutable_device_data<void>()
        ));
    }

    ~Conv2DGradientFilter(){
        cudnnDestroyTensorDescriptor(_x_desc);
        cudnnDestroyFilterDescriptor(_dw_desc);
        cudnnDestroyTensorDescriptor(_dy_desc);
        cudnnDestroyConvolutionDescriptor(_conv_desc);
        cudaFree(_ws_filter);
        cudnnDestroy(_handle);
    }

private:
    Tensor x;
    Tensor dy;
    Tensor dw;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
    void *_ws_filter;
    size_t _ws_filter_size;
    cudnnHandle_t _handle;
    cudnnConvolutionFwdAlgo_t _conv_algo;
    cudnnConvolutionBwdFilterAlgo_t _filter_algo;
    cudnnTensorDescriptor_t _x_desc;
    cudnnFilterDescriptor_t _dw_desc;
    cudnnTensorDescriptor_t _dy_desc;
    cudnnConvolutionDescriptor_t _conv_desc;
};

REGIST_OP_GRAD_ALGO(Conv2DGradientFilter)
    .Input("X", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dW", type::float32::string)
    .Device("CUDA(CUDNN)")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = Conv2DGradientFilter<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cudnn
} // end namespace mlfe
