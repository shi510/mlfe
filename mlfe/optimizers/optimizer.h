#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#include <memory>
#include "mlfe/core/tensor.h"

namespace mlfe{
namespace opt{

class optimizer{
public:
    optimizer(double lr);

    virtual void apply(Tensor var, Tensor var_grad) = 0;

    void update_learning_rate(double lr);

    double get_learning_rate();

protected:
    Tensor _lr;
};

using optimizer_ptr = std::shared_ptr<optimizer>;

} // end namespace optimizer
} // end namespace mlfe
#endif // end ifndef __OPTIMIZER_HPP__
