#include "reshape.h"
#include "../core/gradient_helper.h"

namespace mlfe{ namespace functional{

class ReshapeGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x = y.get_children()[0];
        Tensor dx = functional::reshape(dy, x.shape());
        in_grads.push_back(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(Reshape, ReshapeGradient)

} // end namespace functional
} // end namespace mlfe
