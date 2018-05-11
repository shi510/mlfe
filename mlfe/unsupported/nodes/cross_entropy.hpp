#ifndef __CROSS_ENTROPY_HPP__
#define __CROSS_ENTROPY_HPP__

#include "../core/node.hpp"
#include "../core/tensor.hpp"

namespace mlfe { namespace node {

class SoftmaxCrossEntropyWithLabel : public NodeSchema<SoftmaxCrossEntropyWithLabel>{
public:
    SoftmaxCrossEntropyWithLabel();

protected:
    void InternalInit(Workspace *ws, OperatorContext *oc) override;

    void InternalGradientInit(Workspace *ws, OperatorContext *oc) override;
};

class SigmoidCrossEntropy : public NodeSchema<SigmoidCrossEntropy> {
public:
    SigmoidCrossEntropy();

protected:
    void InternalInit(Workspace *ws, OperatorContext *oc) override;

    void InternalGradientInit(Workspace *ws, OperatorContext *oc) override;
};

} // end namespace node
} // end namespace mlfe
#endif // end ifndef __CROSS_ENTROPY_HPP__
