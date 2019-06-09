#include "math.h"
#include "../core/op_algo.h"
#include "../core/gradient_helper.h"
#include "../operators/basic_arithmetics.h"
#include "../operators/initializer.h"

namespace mlfe{

class SquaredDifferenceGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        namespace fn = functional;
        VecTensor in_grads;
        auto x1 = y.get_children()[0];
        auto x2 = y.get_children()[1];
        auto two = functional::constant(2, x1.shape());
        Tensor dx1 = fn::mul(dy, fn::mul(two, fn::sub(x1, x2)));
        Tensor dx2 = fn::mul(dy, fn::negative(fn::mul(two, fn::sub(x1, x2))));
        in_grads.push_back(dx1);
        in_grads.push_back(dx2);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(SquaredDifference, SquaredDifferenceGradient)

class ReduceMeanGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x = y.get_children()[0];
        Tensor dx = functional::create_variable(x.shape());
        OpAlgoContext ctx("ReduceMeanGradient");
        dx.add_child(dy);
        Tensor::AssignOpFunctor(dx, ctx);
        in_grads.push_back(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(ReduceMean, ReduceMeanGradient)

namespace functional{

Tensor squared_difference(Tensor x1, Tensor x2){
    Tensor y = create_variable(x1.shape());
    OpAlgoContext ctx("SquaredDifference");
    y.add_child(x1);
    y.add_child(x2);
    Tensor::AssignOpFunctor(y, ctx);
    return y;
}

Tensor mean(Tensor x){
    Tensor y = create_variable({1});
    OpAlgoContext ctx("ReduceMean");
    
    y.add_child(x);
    Tensor::AssignOpFunctor(y, ctx);
    
    return y;
}

} // end namespace functional
} // end namespace mlfe
