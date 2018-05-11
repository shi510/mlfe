#ifndef __FILLER_OP_HPP__
#define __FILLER_OP_HPP__
#include "../core/node.hpp"
#include "../../math/functions.hpp"
#include <random>

namespace mlfe { namespace node {

class ConstantInit final : public NodeSchema<ConstantInit> {
public:
    ConstantInit();

protected:
    void InternalInit(Workspace *ws, OperatorContext *oc) override;

    void InternalGradientInit(Workspace *ws, OperatorContext *oc) override;
};

class XavierInit final : public NodeSchema<XavierInit> {
public:
    XavierInit();

protected:
    void InternalInit(Workspace *ws, OperatorContext *oc) override;

    void InternalGradientInit(Workspace *ws, OperatorContext *oc) override;
};

} // end namespace node
} // end namespace mlfe
#endif // end ifndef __FILLER_OP_HPP__
