#include "../core/node.hpp"
#include "../../device_context/cpu_context.hpp"
#include "../../utils/assert.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CPUContext>
struct ReshapeCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override { }

    void Run() override { }
};

REGIST_NODE_FUNCTOR(Reshape, DataType::F32, Accelerator::Default, ReshapeCpuF<float>)

template <typename T, typename D = CPUContext>
struct ReshapeGradCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override { }

    void Run() override { }
};

REGIST_NODE_GRADIENT_FUNCTOR(Reshape, DataType::F32, Accelerator::Default, ReshapeGradCpuF<float>)

} // end namespace node
} // end namespace mlfe