#include "reshape.h"
#include "../core/gradient_helper.h"
#include "mlfe/core/op_algo.h"
#include "mlfe/operators/initializer.h"

namespace mlfe{ namespace functional{

class ReshapeGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x = y.get_context().get_input(0);
        Tensor shape_t = y.get_context().get_input(1);
        Tensor dx = functional::reshape(dy, x.shape());
        auto dshape_t = functional::constant(1, shape_t.shape());
        x.set_backprop_node(dx.get_node());
        x.set_gradient(dx);
        shape_t.set_backprop_node(dshape_t.get_node());
        shape_t.set_gradient(dshape_t);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(Reshape, ReshapeGradient)

} // end namespace functional
} // end namespace mlfe
