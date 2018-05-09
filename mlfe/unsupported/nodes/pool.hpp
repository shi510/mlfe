#ifndef __POOL_OP_HPP__
#define __POOL_OP_HPP__
#include "../core/node.hpp"

namespace mlfe { namespace node {

class MaxPool final : public NodeIO<MaxPool> {
public:
    MaxPool();

protected:
    void InternalInit(Workspace *ws, OperatorContext *oc) override;

    void InternalGradientInit(Workspace *ws, OperatorContext *oc) override;
};

} // end namespace node
} // end namespace mlfe
#endif // end ifndef __POOL_OP_HPP__
