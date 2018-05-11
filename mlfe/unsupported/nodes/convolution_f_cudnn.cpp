#include <cudnn.h>
#include "../core/node.hpp"
#include "../../device_context/cuda_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions.hpp"
#include "../../utils/assert.hpp"
#include "../../math/transform.hpp"
#include "../../math/functions_cuda.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CUDAContext>
struct ConvCudnnF : NodeFunctor {
    void inline ASSERT_SUCCESS(cudnnStatus_t t) {
        if (t != CUDNN_STATUS_SUCCESS) {
            std::cout << "NOT CUDNN_STATUS_SUCCESS" << " : "<< cudnnGetErrorString(t) << std::endl;
            exit(0);
        }
    }
    void Init(OperatorContext *oc) override {
        DataType dt = DataType::F32;
        _kernel = oc->attr->GetParam<std::vector<int>>("Kernel");
        _stride = oc->attr->GetParam<std::vector<int>>("Stride");
        _padding = oc->attr->GetParam<std::vector<int>>("Padding");
        _x = oc->inputs[0];
        _w = oc->inputs[1];
        _b = oc->inputs[2];
        _y = oc->outputs[0];

        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        _w->Allocate(Accelerator::CUDA, dt);
        _b->Allocate(Accelerator::CUDA, dt);
        _y->Allocate(Accelerator::CUDA, dt);

        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_x_desc);
        cudnnCreateFilterDescriptor(&_w_desc);
        cudnnCreateTensorDescriptor(&_b_desc);
        cudnnCreateTensorDescriptor(&_y_desc);
        cudnnCreateConvolutionDescriptor(&_conv_desc);
        cudnnCreateActivationDescriptor(&_act_desc);

        cudnnSetTensor4dDescriptor(
            _x_desc, 
            CUDNN_TENSOR_NCHW,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            _x->Dim(0), _x->Dim(1), _x->Dim(2), _x->Dim(3));

        cudnnSetFilter4dDescriptor(
            _w_desc,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            CUDNN_TENSOR_NCHW,
            _w->Dim(0), _w->Dim(1), _w->Dim(2), _w->Dim(3));

        cudnnSetTensor4dDescriptor(
            _b_desc, 
            CUDNN_TENSOR_NCHW,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            1, _b->Size(), 1, 1);

        cudnnSetTensor4dDescriptor(
            _y_desc, 
            CUDNN_TENSOR_NCHW,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            _y->Dim(0), _y->Dim(1), _y->Dim(2), _y->Dim(3));

        cudnnSetActivationDescriptor(            _act_desc,             CUDNN_ACTIVATION_IDENTITY,             CUDNN_NOT_PROPAGATE_NAN,             1.        );                ASSERT_SUCCESS(cudnnSetConvolution2dDescriptor(
            _conv_desc,
            _padding[0], _padding[1],
            _stride[0], _stride[1], 1, 1,
            CUDNN_CROSS_CORRELATION,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE
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

    void Run() override {
        const float alpha = 1, beta = 0;
        ASSERT_SUCCESS(cudnnConvolutionBiasActivationForward(
            _handle,
            &alpha,
            _x_desc, _x->GetPtr<T>(),
            _w_desc, _w->GetPtr<T>(),
            _conv_desc, _conv_algo,
            _ws_fwd, _ws_fwd_size,
            &beta,
            _y_desc, _y->GetPtr<T>(),
            _b_desc, _b->GetPtr<T>(),
            _act_desc,
            _y_desc, _y->GetPtr<T>()
        ));
    }

    ~ConvCudnnF(){
        cudnnDestroyTensorDescriptor(_x_desc);
        cudnnDestroyFilterDescriptor(_w_desc);
        cudnnDestroyTensorDescriptor(_b_desc);
        cudnnDestroyTensorDescriptor(_y_desc);
        cudnnDestroyConvolutionDescriptor(_conv_desc);
        cudaFree(_ws_fwd);
        cudnnDestroy(_handle);
    }
    
    Tensor *_x, *_w, *_b;
    Tensor *_y;
    std::vector<int> _kernel;
    std::vector<int> _stride;
    std::vector<int> _padding;
    void *_ws_fwd;
    size_t _ws_fwd_size = 0;
    cudnnHandle_t _handle;
    cudnnTensorDescriptor_t _x_desc;
    cudnnFilterDescriptor_t _w_desc;
    cudnnTensorDescriptor_t _b_desc;
    cudnnTensorDescriptor_t _y_desc;
    cudnnActivationDescriptor_t _act_desc;
    cudnnConvolutionFwdAlgo_t _conv_algo;
    cudnnConvolutionDescriptor_t _conv_desc;
};

REGIST_NODE_FUNCTOR(Conv, DataType::F32, Accelerator::CUDNN, ConvCudnnF<float>)
REGIST_NODE_FUNCTOR(Conv, DataType::F64, Accelerator::CUDNN, ConvCudnnF<double>)

template <typename T, typename D = CUDAContext>
struct ConvGradCudnnF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        _kernel = oc->attr->GetParam<std::vector<int>>("Kernel");
        _stride = oc->attr->GetParam<std::vector<int>>("Stride");
        _padding = oc->attr->GetParam<std::vector<int>>("Padding");
        _x = oc->inputs[0];
        _w = oc->inputs[1];
        _dy = oc->inputs[2];
        _dw = oc->outputs[0];
        _db = oc->outputs[1];
        _dx = oc->outputs[2];

        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        _dw->Allocate(Accelerator::CUDA, dt);
        _db->Allocate(Accelerator::CUDA, dt);
        _dx->Allocate(Accelerator::CUDA, dt);

        cudnnCreate(&_handle);
        cudnnCreateTensorDescriptor(&_dx_desc);
        cudnnCreateFilterDescriptor(&_dw_desc);
        cudnnCreateTensorDescriptor(&_db_desc);
        cudnnCreateTensorDescriptor(&_dy_desc);
        cudnnCreateConvolutionDescriptor(&_conv_desc);

        cudnnSetTensor4dDescriptor(_dx_desc, CUDNN_TENSOR_NCHW,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            _x->Dim(0), _x->Dim(1), _x->Dim(2), _x->Dim(3));

        cudnnSetFilter4dDescriptor(_dw_desc,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            CUDNN_TENSOR_NCHW,
            _w->Dim(0), _w->Dim(1), _w->Dim(2), _w->Dim(3));

        cudnnSetTensor4dDescriptor(_db_desc, CUDNN_TENSOR_NCHW,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            1, _db->Size(), 1, 1);

        cudnnSetTensor4dDescriptor(_dy_desc, CUDNN_TENSOR_NCHW,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE,
            _dy->Dim(0), _dy->Dim(1), _dy->Dim(2), _dy->Dim(3));
        cudnnSetConvolution2dDescriptor(
            _conv_desc,
            _padding[0], _padding[1],
            _stride[0], _stride[1], 1, 1,
            CUDNN_CROSS_CORRELATION,
            dt == DataType::F32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE
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

    void Run() override {
        const float alpha = 1, beta = 0;

        cudnnConvolutionBackwardFilter(
            _handle,
            &alpha, _dx_desc, _x->GetPtr<T>(),
            _dy_desc, _dy->GetPtr<T>(),
            _conv_desc, _filter_algo,
            _ws_filter, _ws_filter_size,
            &beta, 
            _dw_desc, _dw->GetPtr<T>()
            );

        cudnnConvolutionBackwardData(
            _handle,
            &alpha,
            _dw_desc, _w->GetPtr<T>(),
            _dy_desc, _dy->GetPtr<T>(),
            _conv_desc, _data_algo,
            _ws_data, _ws_data_size,
            &beta, 
            _dx_desc, _dx->GetPtr<T>()
        );

        cudnnConvolutionBackwardBias(
            _handle,
            &alpha,
            _dy_desc, _dy->GetPtr<T>(),
            &beta,
            _db_desc, _db->GetPtr<T>()
        );
        
    }

    ~ConvGradCudnnF() {
        cudnnDestroyTensorDescriptor(_dx_desc);
        cudnnDestroyFilterDescriptor(_dw_desc);
        cudnnDestroyTensorDescriptor(_db_desc);
        cudnnDestroyTensorDescriptor(_dy_desc);
        cudnnDestroyConvolutionDescriptor(_conv_desc);
        cudaFree(_ws_data);
        cudaFree(_ws_filter);
        cudnnDestroy(_handle);
    }
    
    Tensor *_x, *_w, *_dy;
    Tensor *_dw, *_db, *_dx;
    std::vector<int> _kernel;
    std::vector<int> _stride;
    std::vector<int> _padding;
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
    cudnnTensorDescriptor_t _db_desc;
    cudnnTensorDescriptor_t _dy_desc;
    cudnnConvolutionDescriptor_t _conv_desc;
};

REGIST_NODE_GRADIENT_FUNCTOR(Conv, DataType::F32, Accelerator::CUDNN, ConvGradCudnnF<float>)
REGIST_NODE_GRADIENT_FUNCTOR(Conv, DataType::F64, Accelerator::CUDNN, ConvGradCudnnF<double>)

} // end namespace node
} // end namespace mlfe