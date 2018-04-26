#include "../core/node.hpp"
#include "../../device_context/cpu_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions.hpp"
#include "../../utils/assert.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CPUContext>
struct OneHotCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        _label = oc->inputs[0];
        _onehot = oc->outputs[0];
        _batch = _onehot->Dim(0);
        _classes = _onehot->Dim(1);
        _onehot->Allocate(Accelerator::Default, dt);
    }

    void Run() override {
        math::set<T, D>(
            _onehot->Size(),
            static_cast<T>(0),
            _onehot->GetPtr<T>()
            );
        for (int b = 0; b < _batch; ++b) {
            int label_val = static_cast<int>(_label->GetPtr<T>()[b]);
            _onehot->GetPtr<T>()[b * _classes + label_val] = static_cast<T>(1);
        }
    }

    int _batch;
    int _classes;
    Tensor *_label;
    Tensor *_onehot;
};

REGIST_NODE_FUNCTOR(OneHot, DataType::F32, Accelerator::Default, OneHotCpuF<float>)

template <typename T, typename D = CPUContext>
struct OneHotGradCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {}

    void Run() override {}
};

REGIST_NODE_GRADIENT_FUNCTOR(OneHot, DataType::F32, Accelerator::Default, OneHotGradCpuF<float>)

} // end namespace node
} // end namespace mlfe