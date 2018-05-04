#ifndef __RESHAPE_OP_HPP__
#define __RESHAPE_OP_HPP__
#include "../core/node.hpp"

namespace mlfe { namespace node {

class Reshape final : public NodeIO<Reshape> {
public:
    Reshape();

protected:
    void InternalInit(Workspace *ws, OperatorContext *oc) override;

    void InternalGradientInit(Workspace *ws, OperatorContext *oc) override;
};

} // end namespace node
} // end namespace mlfe
#endif // end ifndef __RESHAPE_OP_HPP__
