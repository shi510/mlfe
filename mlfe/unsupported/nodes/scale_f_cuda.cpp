#include <random>
#include "../core/node.hpp"
#include "../../device_context/cuda_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions.hpp"
#include "../../utils/assert.hpp"
#include "../../math/functions_cuda.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CUDAContext>
struct ScaleCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        auto val_str = oc->attr->GetParam<std::string>("Value");
        _val = to_value<T>(val_str);
        _x = oc->inputs[0];
        _y = oc->outputs[0];
        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        if (_y->GetPtr<T>() == nullptr) {
            _y->Allocate(Accelerator::CUDA, dt);
        }
    }

    void Run() override {
        math::scal<T, D>(
            _x->Size(),
            _val,
            _x->GetPtr<T>(),
            _y->GetPtr<T>()
            );
    }

    Tensor *_x;
    Tensor *_y;
    T _val;
};

REGIST_NODE_FUNCTOR(Scale, DataType::F32, Accelerator::CUDA, ScaleCudaF<float>)
REGIST_NODE_FUNCTOR(Scale, DataType::F64, Accelerator::CUDA, ScaleCudaF<double>)

template <typename T, typename D = CUDAContext>
struct ScaleGradCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        auto val_str = oc->attr->GetParam<std::string>("Value");
        _val = to_value<T>(val_str);
        _dy = oc->inputs[0];
        _dx = oc->outputs[0];
        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        if (_dx->GetPtr<T>() == nullptr) {
            _dx->Allocate(Accelerator::CUDA, dt);
        }
    }

    void Run() override {
        math::scal<T, D>(
            _dx->Size(),
            _val,
            _dy->GetPtr<T>(),
            _dx->GetPtr<T>()
            );
    }

    Tensor *_dy;
    Tensor *_dx;
    T _val;
};

REGIST_NODE_GRADIENT_FUNCTOR(Scale, DataType::F32, Accelerator::CUDA, ScaleGradCudaF<float>)
REGIST_NODE_GRADIENT_FUNCTOR(Scale, DataType::F64, Accelerator::CUDA, ScaleGradCudaF<double>)

} // end namespace node
} // end namespace mlfe