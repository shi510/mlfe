#include "../core/op_algo.h"
#include "../core/device.h"
#include "../core/tensor_mem_ref.h"
#include "../math/blas.h"
#include "../math/transform.h"
#include "../device_context/cuda_context.h"
#include <cudnn.h>
#include <string>

namespace mlfe{ namespace algorithm_cudnn{

template <class Dev, class Tp>
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
    Convolution(OpAlgoContext *oac) : OpAlgo(oac){
        using IntVec = std::vector<type::int32::T>;
        x = oac->get_input(0);
        w = oac->get_input(1);
        y = oac->get_output(0);
        filters = oac->GetAttr<type::int32::T>("filters");
        filters_hw = oac->GetAttr<IntVec>("filters_hw");
        strides = oac->GetAttr<IntVec>("strides");
        pads = oac->GetAttr<IntVec>("pads");
        auto cudnn_type = [](std::string type_str){
            return type_str == type::float32::string ?
                CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
        };

        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_x_desc);
        cudnnCreateFilterDescriptor(&_w_desc);
        cudnnCreateTensorDescriptor(&_y_desc);
        cudnnCreateConvolutionDescriptor(&_conv_desc);

        cudnnSetTensor4dDescriptor(
            _x_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            x->Shape()[0], x->Shape()[1], x->Shape()[2], x->Shape()[3]);

        cudnnSetFilter4dDescriptor(
            _w_desc,
            cudnn_type(Tp::string),
            CUDNN_TENSOR_NCHW,
            w->Shape()[0], w->Shape()[1], w->Shape()[2], w->Shape()[3]);

        cudnnSetTensor4dDescriptor(
            _y_desc,
            CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            y->Shape()[0], y->Shape()[1], y->Shape()[2], y->Shape()[3]);

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
            _x_desc, x->Data<T>(),
            _w_desc, w->Data<T>(),
            _conv_desc, _conv_algo,
            _ws_fwd, _ws_fwd_size,
            &beta,
            _y_desc, y->Data<T>()
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
    TensorMemRef *x;
    TensorMemRef *w;
    TensorMemRef *y;
    type::int32::T filters;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
    void *_ws_fwd;
    size_t _ws_fwd_size = 0;
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
    .Device(Device::CUDA::string_cudnn)
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Convolution<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class ConvolutionGrad : public OpAlgo{
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

    ConvolutionGrad(OpAlgoContext *oac) : OpAlgo(oac){
        using IntVec = std::vector<type::int32::T>;
        x = oac->get_input(0);
        w = oac->get_input(1);
        y = oac->get_input(2);
        dy = oac->get_input(3);
        dw = oac->get_output(0);
        dx = oac->get_output(1);
        filters = oac->GetAttr<type::int32::T>("filters");
        filters_hw = oac->GetAttr<IntVec>("filters_hw");
        strides = oac->GetAttr<IntVec>("strides");
        pads = oac->GetAttr<IntVec>("pads");
        auto cudnn_type = [](std::string type_str){
            return type_str == type::float32::string ?
                CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
        };

        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_dx_desc);
        cudnnCreateFilterDescriptor(&_dw_desc);
        cudnnCreateTensorDescriptor(&_dy_desc);
        cudnnCreateConvolutionDescriptor(&_conv_desc);

        cudnnSetTensor4dDescriptor(_dx_desc, CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            x->Shape()[0], x->Shape()[1], x->Shape()[2], x->Shape()[3]);

        cudnnSetFilter4dDescriptor(_dw_desc,
            cudnn_type(Tp::string),
            CUDNN_TENSOR_NCHW,
            w->Shape()[0], w->Shape()[1], w->Shape()[2], w->Shape()[3]);

        cudnnSetTensor4dDescriptor(_dy_desc, CUDNN_TENSOR_NCHW,
            cudnn_type(Tp::string),
            dy->Shape()[0], dy->Shape()[1], dy->Shape()[2], dy->Shape()[3]);

        cudnnSetConvolution2dDescriptor(
            _conv_desc,
            pads[0], pads[1],
            strides[0], strides[1], 1, 1,
            CUDNN_CROSS_CORRELATION,
            cudnn_type(Tp::string)
        );

        cudnnGetConvolutionForwardAlgorithm(
            _handle,
            _dx_desc,
            _dw_desc,
            _conv_desc,
            _dy_desc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &_conv_algo
        );

        cudnnGetConvolutionBackwardDataAlgorithm(
            _handle,
            _dw_desc,
            _dy_desc,
            _conv_desc,
            _dx_desc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &_data_algo
        );

        cudnnGetConvolutionBackwardFilterAlgorithm(
            _handle,
            _dx_desc,
            _dy_desc,
            _conv_desc,
            _dw_desc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &_filter_algo
        );

        cudnnGetConvolutionBackwardDataWorkspaceSize(
            _handle,
            _dw_desc,
            _dy_desc,
            _conv_desc,
            _dx_desc,
            _data_algo,
            &_ws_data_size
        );

        cudnnGetConvolutionBackwardFilterWorkspaceSize(
            _handle,
            _dx_desc,
            _dy_desc,
            _conv_desc,
            _dw_desc,
            _filter_algo,
            &_ws_filter_size
        );

        cudaMalloc(&_ws_data, _ws_data_size);
        cudaMalloc(&_ws_filter, _ws_filter_size);
    }

    void Compute() override{
        const float alpha = 1, beta = 0;
        ASSERT_SUCCESS(cudnnConvolutionBackwardFilter(
            _handle,
            &alpha, _dx_desc, x->Data<T>(),
            _dy_desc, dy->Data<T>(),
            _conv_desc, _filter_algo,
            _ws_filter, _ws_filter_size,
            &beta,
            _dw_desc, dw->Data<T>()
        ));

        ASSERT_SUCCESS(cudnnConvolutionBackwardData(
            _handle,
            &alpha,
            _dw_desc, w->Data<T>(),
            _dy_desc, dy->Data<T>(),
            _conv_desc, _data_algo,
            _ws_data, _ws_data_size,
            &beta,
            _dx_desc, dx->Data<T>()
        ));
    }

    ~ConvolutionGrad(){
        cudnnDestroyTensorDescriptor(_dx_desc);
        cudnnDestroyFilterDescriptor(_dw_desc);
        cudnnDestroyTensorDescriptor(_dy_desc);
        cudnnDestroyConvolutionDescriptor(_conv_desc);
        cudaFree(_ws_data);
        cudaFree(_ws_filter);
        cudnnDestroy(_handle);
    }

private:
    TensorMemRef *x;
    TensorMemRef *w;
    TensorMemRef *y;
    TensorMemRef *dy;
    TensorMemRef *dw;
    TensorMemRef *dx;
    type::int32::T filters;
    std::vector<type::int32::T> filters_hw;
    std::vector<type::int32::T> strides;
    std::vector<type::int32::T> pads;
    void *_ws_data;
    void *_ws_filter;
    size_t _ws_data_size;
    size_t _ws_filter_size;
    cudnnHandle_t _handle;
    cudnnConvolutionFwdAlgo_t _conv_algo;
    cudnnConvolutionBwdDataAlgo_t _data_algo;
    cudnnConvolutionBwdFilterAlgo_t _filter_algo;
    cudnnTensorDescriptor_t _dx_desc;
    cudnnFilterDescriptor_t _dw_desc;
    cudnnTensorDescriptor_t _dy_desc;
    cudnnConvolutionDescriptor_t _conv_desc;
};

REGIST_OP_GRAD_ALGO(Convolution)
    .Input("X", type::float32::string)
    .Input("W", type::float32::string)
    .Input("Y", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dW", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CUDA::string_cudnn)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ConvolutionGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cudnn
} // end namespace mlfe
