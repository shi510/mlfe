#include "../core/node.hpp"
#include "../../device_context/cuda_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions.hpp"
#include "../../math/functions_cuda.hpp"
#include "../../utils/assert.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CUDAContext>
struct AccuracyCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        _top_k = oc->attr->GetParam<int>("Top");
        _prob = oc->inputs[0];
        _label = oc->inputs[1];
        _accuracy = oc->outputs[0];
        _batch = _prob->Dim(0);
        _classes = _prob->Dim(1);
        _accuracy->Allocate(Accelerator::CUDA, dt);
    }

    void Run() override {
        math::AccuracyCuda<T>(
            _batch,
            _classes,
            _top_k, 
            _prob->GetPtr<T>(),
            _label->GetPtr<T>(), 
            _accuracy->GetPtr<T>()
            );
    }

    int _batch;
    int _classes;
    Tensor *_prob, *_label;
    Tensor *_accuracy;
    T _top_k;
};

REGIST_NODE_FUNCTOR(Accuracy, DataType::F32, Accelerator::CUDA, AccuracyCudaF<float>)

template <typename T, typename D = CUDAContext>
struct AccuracyGradCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override {}

    void Run() override {}
};

REGIST_NODE_GRADIENT_FUNCTOR(Accuracy, DataType::F32, Accelerator::CUDA, AccuracyGradCudaF<float>)

} // end namespace node
} // end namespace mlfe