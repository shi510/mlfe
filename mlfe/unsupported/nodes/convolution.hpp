#ifndef __CONVOLUTION_OP_HPP__
#define __CONVOLUTION_OP_HPP__
#include "../core/node.hpp"

namespace mlfe { namespace node {

class Conv final : public NodeIO<Conv> {
public:
    Conv();

protected:
    void InternalInit(Workspace *ws, OperatorContext *oc) override;

    void InternalGradientInit(Workspace *ws, OperatorContext *oc) override;
};

} // end namespace node
} // end namespace mlfe
#endif // end ifndef __CONVOLUTION_OP_HPP__
