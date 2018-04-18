#ifndef __DENSE_HPP__
#define __DENSE_HPP__

#include "../core/node.hpp"
#include "../core/tensor.hpp"

namespace mlfe { namespace node {

class Dense : public NodeIO<Dense> {
public:
    Dense();

protected:
    void InternalInit(Workspace *ws, OperatorContext *oc) override;

    void InternalGradientInit(Workspace *ws, OperatorContext *oc) override;
};

} // end namespace node
} // end namespace mlfe
#endif // end ifndef __DENSE_HPP__
