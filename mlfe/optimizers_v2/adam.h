#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/op_kernel.h"
#include <functional>
#include <vector>

namespace mlfe{
using namespace operators_v2;
using adam_fn_t = std::function<void (Tensor x, Tensor dx, Tensor m_hist, Tensor v_hist, float lr, float beta1, float beta2, float eps)>;
DECLARE_OP_KERNEL(adam, adam_fn_t);

namespace optimizers{

class adam
{
public:
    adam(float lr, float beta1=0.9f, float beta2=0.999f, float eps=1e-5f);

    void set_variables(std::vector<Tensor> vars);

    void update();

private:
    float __lr;
    float __beta1;
    float __beta2;
    float __eps;
    std::vector<Tensor> __vars;
    std::vector<Tensor> __m_hist;
    std::vector<Tensor> __v_hist;
};

} // end namespace optimizers
} // end namespace mlfe
