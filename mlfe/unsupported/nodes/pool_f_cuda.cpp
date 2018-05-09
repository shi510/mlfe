#include "../core/node.hpp"
#include "../../device_context/cuda_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions.hpp"
#include "../../utils/assert.hpp"
#include "../../math/functions_cuda.hpp"
#include "../../math/transform.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CUDAContext>
struct MaxPoolCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        _kernel = oc->attr->GetParam<std::vector<int>>("Kernel");
        _stride = oc->attr->GetParam<std::vector<int>>("Stride");
        _x = oc->inputs[0];
        _y = oc->outputs[0];
        _idx = oc->outputs[1];
        _batch = _x->Dim(0);
        _in_c = _x->Dim(1);
        _in_h = _x->Dim(2);
        _in_w = _x->Dim(3);
        _out_h = _y->Dim(2);
        _out_w = _y->Dim(3);

        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }

        _y->Allocate(Accelerator::CUDA, dt);
        _idx->Allocate(Accelerator::CUDA, dt);
    }

    void Run() override {
        math::MaxPool<T, D>(
            _y->Size(), _x->GetPtr<T>(),
            _in_c, _in_h, _in_w, _y->Dim(2), _y->Dim(3),
            _kernel[0], _kernel[1], _stride[0], _stride[1],
            0, 0, 
            _y->GetPtr<T>(), _idx->GetPtr<int>()
            );
    }

    Tensor *_x;
    Tensor *_idx, *_y;
    int _batch;
    int _in_c;
    int _in_h, _in_w;
    int _out_h, _out_w;
    std::vector<int> _kernel;
    std::vector<int> _stride;
};

REGIST_NODE_FUNCTOR(MaxPool, DataType::F32, Accelerator::CUDA, MaxPoolCudaF<float>)
//REGIST_NODE_FUNCTOR(MaxPool, DataType::F64, Accelerator::CUDA, MaxPoolCudaF<double>)

template <typename T, typename D = CUDAContext>
struct MaxPoolGradCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        _kernel = oc->attr->GetParam<std::vector<int>>("Kernel");
        _stride = oc->attr->GetParam<std::vector<int>>("Stride");
        _idx = oc->inputs[2];
        _dy = oc->inputs[3];
        _dx = oc->outputs[0];
        _batch = _dy->Dim(0);
        _in_c = _dx->Dim(1);
        _in_h = _dx->Dim(2);
        _in_w = _dx->Dim(3);
        _out_h = _idx->Dim(2);
        _out_w = _idx->Dim(3);
        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        _dx->Allocate(Accelerator::CUDA, dt);
    }

    void Run() override {
        math::MaxPoolGradient<T, D>(
            _dy->Size(),
            _dy->GetPtr<T>(), _idx->GetPtr<int>(), 
            _in_c, _in_h, _in_w, 
            _dy->Dim(2), _dy->Dim(3), 
            _kernel[0], _kernel[1], 
            _stride[0], _stride[1], 
            0, 0, 
            _dx->GetPtr<T>()
            );
    }

    Tensor *_idx, *_dy;
    Tensor *_dx;
    int _batch;
    int _in_c;
    int _in_h, _in_w;
    int _out_h, _out_w;
    std::vector<int> _kernel;
    std::vector<int> _stride;
};

REGIST_NODE_GRADIENT_FUNCTOR(MaxPool, DataType::F32, Accelerator::CUDA, MaxPoolGradCudaF<float>)
//REGIST_NODE_GRADIENT_FUNCTOR(MaxPool, DataType::F64, Accelerator::CUDA, MaxPoolGradCudaF<double>)

} // end namespace node
} // end namespace mlfe