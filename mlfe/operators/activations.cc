#include "activations.h"
#include "../core/op_algo.h"
#include "../core/tensor.h"
#include "../core/gradient_helper.h"

namespace mlfe{ namespace functional{

class ReLUGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x = y.get_children()[0];
        Tensor dx = create_variable(x.shape());
        OpAlgoContext cxt("ReLUGradient");
        dx.add_child(x);
        dx.add_child(y);
        dx.add_child(dy);
        Tensor::AssignOpFunctor(dx, cxt);
        in_grads.push_back(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(ReLU, ReLUGradient)

class SigmoidGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x = y.get_children()[0];
        Tensor dx = create_variable(x.shape());
        OpAlgoContext cxt("SigmoidGradient");
        dx.add_child(x);
        dx.add_child(y);
        dx.add_child(dy);
        Tensor::AssignOpFunctor(dx, cxt);
        in_grads.push_back(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(Sigmoid, SigmoidGradient)

Tensor relu(Tensor x){
    Tensor y = functional::create_variable(x.shape());
    OpAlgoContext cxt("ReLU");
    y.add_child(x);
    Tensor::AssignOpFunctor(y, cxt);

    return y;
}

Tensor sigmoid(Tensor x){
    Tensor y = functional::create_variable(x.shape());
    OpAlgoContext cxt("Sigmoid");
    y.add_child(x);
    Tensor::AssignOpFunctor(y, cxt);

    return y;
}

} // end namespace functional
} // end namespace mlfe
