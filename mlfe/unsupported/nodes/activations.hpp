#ifndef __ACTIVATIONS_OP_HPP__
#define __ACTIVATIONS_OP_HPP__
#include "../core/node.hpp"

namespace mlfe { namespace node {

class Activation final : public NodeSchema<Activation> {
public:
    Activation();

protected:
    void InternalInit(Workspace *ws, OperatorContext *oc) override;

    void InternalGradientInit(Workspace *ws, OperatorContext *oc) override;
};

enum class ActivationType {
    ReLU, Sigmoid
};

} // end namespace node
} // end namespace mlfe
#endif // end ifndef __ACTIVATIONS_OP_HPP__
