#include "pool.h"
#include "mlfe/core/op_algo.h"
#include "mlfe/core/gradient_helper.h"

namespace mlfe{ 

class MaxPoolGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        using Ints = std::vector<type::int32::T>;
        VecTensor in_grads;
        auto y_ctx = y.get_context();
        Tensor x = y_ctx.get_input(0);
        Tensor dx = functional::create_variable(x.shape());
        OpAlgoContext ctx("MaxPoolGradient");
        ctx.add_input(x);
        ctx.add_input(y);
        ctx.add_input(dy);
        ctx.add_output(dx);
        ctx.set_attrs(y_ctx.get_attrs());
        dx.set_context(ctx);
        x.set_backprop_node(dx.get_node());
        x.set_gradient(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(MaxPool, MaxPoolGradient)

class AvgPoolGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        using IntVec = std::vector<type::int32::T>;
        VecTensor in_grads;

        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(AvgPool, AvgPoolGradient)

namespace functional{

Tensor pool_max(Tensor x, 
                std::vector<int> kernel, 
                std::vector<int> stride, 
                std::vector<int> padding
               ){
    Tensor y;
    OpAlgoContext ctx("MaxPool");
    ctx.add_input(x);
    ctx.add_output(y);
    ctx.add_attr({"kernel", kernel});
    ctx.add_attr({"stride", stride});
    ctx.add_attr({"padding", padding});
    y.set_context(ctx);
    return y;
}

Tensor global_average_pool(Tensor x){
    Tensor y;
    OpAlgoContext ctx("GlobalAveragePool");
    ctx.add_input(x);
    ctx.add_output(y);
    y.set_context(ctx);
    return y;
}

} // end namespace functional
} // end namespace mlfe
