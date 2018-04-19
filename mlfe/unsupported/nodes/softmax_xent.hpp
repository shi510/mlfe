#ifndef __SOFTMAX_XENT_HPP__
#define __SOFTMAX_XENT_HPP__

#include "../core/node.hpp"
#include "../core/tensor.hpp"

namespace mlfe { namespace node {

class SoftmaxCrossEntropy : public NodeIO<SoftmaxCrossEntropy>{
public:
    SoftmaxCrossEntropy();

protected:
    void InternalInit(Workspace *ws, OperatorContext *oc) override;

    void InternalGradientInit(Workspace *ws, OperatorContext *oc) override;
};

} // end namespace node
} // end namespace mlfe
#endif // end ifndef __SOFTMAX_XENT_HPP__
