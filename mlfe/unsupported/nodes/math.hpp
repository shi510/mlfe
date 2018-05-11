#ifndef __MATH_NODE_HPP__
#define __MATH_NODE_HPP__

#include "../core/node.hpp"
#include "../core/tensor.hpp"

namespace mlfe {namespace node {

#define DECLARE_BIANRY_OP_NODE(OpName)\
class OpName : public NodeSchema<OpName> {\
public:\
    OpName();\
protected:\
    void InternalInit(Workspace *ws, OperatorContext *oc) override;\
    void InternalGradientInit(Workspace *ws, OperatorContext *oc) override;\
};

DECLARE_BIANRY_OP_NODE(Add)
DECLARE_BIANRY_OP_NODE(Sub)
DECLARE_BIANRY_OP_NODE(Mul)
DECLARE_BIANRY_OP_NODE(Div)

} // end namespace node
} // end namespace mlfe
#endif // end ifndef __MATH_NODE_HPP__
