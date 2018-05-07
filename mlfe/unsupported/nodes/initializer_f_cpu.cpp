#include "../core/node.hpp"
#include <random>
#include "../../device_context/cpu_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions.hpp"
#include "../../utils/assert.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CPUContext>
struct ConstantInitCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        _val = oc->attr->GetParam<int>("Value");
        _x = oc->inputs[0];
    }

    void Run() override {
        math::set<T, D>(
            _x->Size(), 
            _val, 
            _x->GetPtr<T>()
            );
    }

    Tensor *_x;
    T _val;
};

REGIST_NODE_FUNCTOR(ConstantInit, DataType::F32, Accelerator::Default, ConstantInitCpuF<float>)

template <typename T, typename D = CPUContext>
struct ConstantInitGradCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override { }

    void Run() override { }
};

REGIST_NODE_GRADIENT_FUNCTOR(ConstantInit, DataType::F32, Accelerator::Default, ConstantInitGradCpuF<float>)

template <typename T>
struct XavierInitCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        int seed = oc->attr->GetParam<int>("Seed");
        _x = oc->inputs[0];
        _rng = std::mt19937(seed);
        T scale = std::sqrt(
            static_cast<T>(6) /
            static_cast<T>(_x->Size() / _x->Dim(0))
        );
        _dist = std::uniform_real_distribution<T>(-scale, scale);
    }

    void Run() override {
        int size = _x->Size();
        T *x_ptr = _x->GetPtr<T>();
        
        for (int n = 0; n < size; ++n) {
            x_ptr[n] = static_cast<T>(_dist(_rng));
        }
    }
    Tensor *_x;
    std::mt19937 _rng;
    std::uniform_real_distribution<T> _dist;
};

REGIST_NODE_FUNCTOR(XavierInit, DataType::F32, Accelerator::Default, XavierInitCpuF<float>)

template <typename T>
struct XavierInitGradCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override { }

    void Run() override { }
};

REGIST_NODE_GRADIENT_FUNCTOR(XavierInit, DataType::F32, Accelerator::Default, XavierInitGradCpuF<float>)

} // end namespace node
} // end namespace mlfe