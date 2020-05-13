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
        auto ctx_y = y.get_context();
        auto x1 = ctx_y.get_input(0);
        auto x2 = ctx_y.get_input(1);
        auto two = functional::constant(2, x1.shape());
        Tensor dx1 = fn::mul(dy, fn::mul(two, fn::sub(x1, x2)));
        Tensor dx2 = fn::mul(dy, fn::negative(fn::mul(two, fn::sub(x1, x2))));
        x1.set_backprop_node(dx1.get_node());
        x1.set_gradient(dx1);
        x2.set_backprop_node(dx2.get_node());
        x2.set_gradient(dx2);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(SquaredDifference, SquaredDifferenceGradient)

class ReduceMeanGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        auto ctx_y = y.get_context();
        Tensor x = ctx_y.get_input(0);
        Tensor dx = functional::create_variable(x.shape());
        OpAlgoContext ctx("ReduceMeanGradient");
        ctx.add_input(dy);
        ctx.add_output(dx);
        dx.set_context(ctx);
        auto op = find_op(ctx);
        dx.get_node().set_task(make_task([](decltype(op) op) {
            op->Compute(); }, op));
        dx.get_node().add_input(dy.get_node());
        dx.get_node().add_input(y.get_node());
        x.set_backprop_node(dx.get_node());
        x.set_gradient(dx);
        dx.get_node().add_attr("op_name", std::string("ReduceMeanGradient"));
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(ReduceMean, ReduceMeanGradient)

namespace functional{

Tensor squared_difference(Tensor x1, Tensor x2){
    Tensor y = create_variable(x1.shape());
    OpAlgoContext ctx("SquaredDifference");
    ctx.add_input(x1);
    ctx.add_input(x2);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

Tensor mean(Tensor x){
    Tensor y = functional::create_variable({ 1 });
    OpAlgoContext ctx("ReduceMean");;
    ctx.add_input(x);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

} // end namespace functional
} // end namespace mlfe
