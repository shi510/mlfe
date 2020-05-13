#include "reshape.h"
#include "../core/gradient_helper.h"
#include "mlfe/core/op_algo.h"

namespace mlfe{ namespace functional{

class ReshapeGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x = y.get_context().get_input(0);
        Tensor dx = functional::reshape(dy, x.shape());
        x.set_backprop_node(dx.get_node());
        x.set_gradient(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(Reshape, ReshapeGradient)

} // end namespace functional
} // end namespace mlfe
