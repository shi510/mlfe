#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#include <memory>

namespace mlfe{
//forward declaration
class Tensor;

namespace opt{

class optimizer{
public:
    virtual void apply(Tensor var, Tensor var_grad) = 0;

protected:
    optimizer();
};

using optimizer_ptr = std::shared_ptr<optimizer>;

} // end namespace optimizer
} // end namespace mlfe
#endif // end ifndef __OPTIMIZER_HPP__
