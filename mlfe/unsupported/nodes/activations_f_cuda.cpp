#include "../core/node.hpp"
#include "../../device_context/cuda_context.hpp"
#include "../../utils/assert.hpp"
#include "../../math/functions.hpp"
#include "activations.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CUDAContext>
struct ReLUCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        _x = oc->inputs[0];
        _y = oc->outputs[0];
        _y->Allocate(Accelerator::CUDA);
        _type = oc->attr->GetParam<ActivationType>("Type");
    }

    void Run() override {
        switch (_type) {
        case ActivationType::ReLU:
            math::ReluFunction<T, D>(
                _x->Size(),
                _x->GetPtr<T>(),
                _y->GetPtr<T>()
                );
            break;
        case ActivationType::Sigmoid:
            math::SigmoidFunction<T, D>(
                _x->Size(),
                _x->GetPtr<T>(),
                _y->GetPtr<T>()
                );
            break;
        }
    }

    Tensor *_x;
    Tensor *_y;
    ActivationType _type;
};

REGIST_NODE_FUNCTOR(Activation, DataType::F32, Accelerator::CUDA, ReLUCudaF<float>)
//REGIST_NODE_FUNCTOR(Activation, DataType::F64, Accelerator::CUDA, ReLUCudaF<double>)

template <typename T, typename D = CUDAContext>
struct ReLUGradCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        _x = oc->inputs[0];
        _y = oc->inputs[1];
        _dy = oc->inputs[2];
        _dx = oc->outputs[0];
        _dx->Allocate(Accelerator::CUDA);
        _type = oc->attr->GetParam<ActivationType>("Type");
    }

    void Run() override {
        switch (_type) {
        case ActivationType::ReLU:
            math::ReluGradientFunction<T, D>(
                _dy->Size(),
                _x->GetPtr<T>(),
                _dy->GetPtr<T>(),
                _dx->GetPtr<T>()
                );
            break;
        case ActivationType::Sigmoid:
            math::SigmoidGradientFunction<T, D>(
                _dy->Size(),
                _y->GetPtr<T>(),
                _dy->GetPtr<T>(),
                _dx->GetPtr<T>()
                );
            break;
        }
    }

    Tensor *_x, *_y, *_dy;
    Tensor *_dx;
    ActivationType _type;
};

REGIST_NODE_GRADIENT_FUNCTOR(Activation, DataType::F32, Accelerator::CUDA, ReLUGradCudaF<float>)
//REGIST_NODE_GRADIENT_FUNCTOR(Activation, DataType::F64, Accelerator::CUDA, ReLUGradCudaF<double>)

} // end namespace node
} // end namespace mlfe