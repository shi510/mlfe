#include "../core/node.hpp"
#include "../../device_context/cuda_context.hpp"
#include "../../utils/assert.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CUDAContext>
struct ReshapeCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override { }

    void Run() override { }
};

REGIST_NODE_FUNCTOR(Reshape, DataType::F32, Accelerator::CUDA, ReshapeCudaF<float>)

template <typename T, typename D = CUDAContext>
struct ReshapeGradCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override { }

    void Run() override { }
};

REGIST_NODE_GRADIENT_FUNCTOR(Reshape, DataType::F32, Accelerator::CUDA, ReshapeGradCudaF<float>)

} // end namespace node
} // end namespace mlfe