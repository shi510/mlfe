#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"
#include <functional>
#include <vector>

namespace mlfe{
using namespace operators;
using sgd_fn_t = std::function<void (Tensor x, Tensor dx, Tensor mm_hist, float lr, float mm, float decay)>;
DECLARE_OP_KERNEL(sgd, sgd_fn_t);

namespace optimizers{

class SGD
{
public:
    SGD(float lr, float momentum=0.f);

    void set_variables(std::vector<Tensor> vars);

    void update();

private:
    float __lr;
    float __mm;
    std::vector<Tensor> __var_moments;
    std::vector<Tensor> __vars;
};

} // end namespace optimizers
} // end namespace mlfe
