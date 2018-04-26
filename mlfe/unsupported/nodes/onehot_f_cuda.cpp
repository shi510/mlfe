#include <random>
#include "../core/node.hpp"
#include "../../device_context/cuda_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions_cuda.hpp"
#include "../../utils/assert.hpp"
#include "../../math/functions_cuda.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CUDAContext>
struct OneHotCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        _label = oc->inputs[0];
        _onehot = oc->outputs[0];
        _batch = _onehot->Dim(0);
        _classes = _onehot->Dim(1);
        _onehot->Allocate(Accelerator::CUDA, dt);
    }

    void Run() override {
        math::set<T, D>(
            _onehot->Size(),
            static_cast<T>(0),
            _onehot->GetPtr<T>()
            );

        math::OneHotCuda<T>(
            _batch,
            _classes, 
            _label->GetPtr<T>(), 
            _onehot->GetPtr<T>()
            );
    }

    int _batch;
    int _classes;
    Tensor *_label;
    Tensor *_onehot;
};

REGIST_NODE_FUNCTOR(OneHot, DataType::F32, Accelerator::CUDA, OneHotCudaF<float>)

template <typename T, typename D = CUDAContext>
struct OneHotGradCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override {}

    void Run() override {}
};

REGIST_NODE_GRADIENT_FUNCTOR(OneHot, DataType::F32, Accelerator::CUDA, OneHotGradCudaF<float>)

} // end namespace node
} // end namespace mlfe