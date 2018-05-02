#include "../core/node.hpp"
#include "../../device_context/cpu_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions.hpp"
#include "../../utils/assert.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CPUContext>
struct AccuracyCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        auto val_str = oc->attr->GetParam<std::string>("Top");
        _top_k = to_value<int>(val_str);
        _prob = oc->inputs[0];
        _label = oc->inputs[1];
        _accuracy = oc->outputs[0];
        _batch = _prob->Dim(0);
        _classes = _prob->Dim(1);
        _accuracy->Allocate(Accelerator::Default, dt);
    }

    int TopCount(T *prob_ptr, int label) {
        int count = 0;
        T prob_true = prob_ptr[label];
        for (int n = 0; n < _classes; ++n) {
            if (prob_ptr[n] > prob_true) {
                ++count;
            }
        }
        return count;
    }

    void Run() override {
        int hit_count = 0;
        for (int b = 0; b < _batch; ++b) {
            int top_count = TopCount(
                _prob->GetPtr<T>() + b * _classes,
                static_cast<int>(_label->GetPtr<T>()[b])
            );
            if (top_count < _top_k) {
                ++hit_count;
            }
        }
        _accuracy->GetPtr<T>()[0] = static_cast<T>(hit_count) / static_cast<T>(_batch);
    }

    int _batch;
    int _classes;
    Tensor *_prob, *_label;
    Tensor *_accuracy;
    T _top_k;
};

REGIST_NODE_FUNCTOR(Accuracy, DataType::F32, Accelerator::Default, AccuracyCpuF<float>)

template <typename T, typename D = CPUContext>
struct AccuracyGradCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {}

    void Run() override {}
};

REGIST_NODE_GRADIENT_FUNCTOR(Accuracy, DataType::F32, Accelerator::Default, AccuracyGradCpuF<float>)

} // end namespace node
} // end namespace mlfe