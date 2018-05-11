#ifndef __ONEHOT_OP_HPP__
#define __ONEHOT_OP_HPP__
#include "../core/node.hpp"

namespace mlfe { namespace node {

class OneHot final : public NodeSchema<OneHot> {
public:
    OneHot();

protected:
    void InternalInit(Workspace *ws, OperatorContext *oc) override;

    void InternalGradientInit(Workspace *ws, OperatorContext *oc) override;
};

} // end namespace node
} // end namespace mlfe
#endif // end ifndef __ONEHOT_OP_HPP__
