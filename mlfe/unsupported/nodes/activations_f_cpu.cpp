#include "../core/node.hpp"
#include "../../device_context/cpu_context.hpp"
#include "../../utils/assert.hpp"
#include "../../math/functions.hpp"
#include "activations.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CPUContext>
struct ActivationCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        _x = oc->inputs[0];
        _y = oc->outputs[0];
        _y->Allocate();
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

REGIST_NODE_FUNCTOR(Activation, DataType::F32, Accelerator::Default, ActivationCpuF<float>)
//REGIST_NODE_FUNCTOR(Activation, DataType::F64, Accelerator::Default, ActivationCpuF<double>)

template <typename T, typename D = CPUContext>
struct ActivationGradCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override { 
        _x = oc->inputs[0];
        _y = oc->inputs[1];
        _dy = oc->inputs[2];
        _dx = oc->outputs[0];
        _dx->Allocate();
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

REGIST_NODE_GRADIENT_FUNCTOR(Activation, DataType::F32, Accelerator::Default, ActivationGradCpuF<float>)
//REGIST_NODE_GRADIENT_FUNCTOR(Activation, DataType::F64, Accelerator::Default, ActivationGradCpuF<double>)

} // end namespace node
} // end namespace mlfe