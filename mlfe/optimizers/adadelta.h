#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"
#include <functional>
#include <vector>

namespace mlfe{
using namespace operators;
using adadelta_fn_t = std::function<void (Tensor, Tensor, Tensor, Tensor, float, float, float)>;
DECLARE_OP_KERNEL(adadelta, adadelta_fn_t);

namespace optimizers{

class adadelta
{
public:
    adadelta(float lr, float momentum, float eps=1e-6f);

    void set_variables(std::vector<Tensor> vars);

    void update();

private:
    float __lr;
    float __momentum;
    float __eps;
    std::vector<Tensor> __vars;
    std::vector<Tensor> __grad_hist;
    std::vector<Tensor> __acc_hist;
};

} // end namespace optimizers
} // end namespace mlfe
