#ifndef __SCALE_OP_HPP__
#define __SCALE_OP_HPP__
#include "../core/node.hpp"

namespace mlfe { namespace node {

class Scale final : public NodeSchema<Scale> {
public:
    Scale();

protected:
    void InternalInit(Workspace *ws, OperatorContext *oc) override;

    void InternalGradientInit(Workspace *ws, OperatorContext *oc) override;
};

} // end namespace node
} // end namespace mlfe
#endif // end ifndef __SCALE_OP_HPP__
