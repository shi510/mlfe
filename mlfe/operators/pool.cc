#include "pool.h"
#include "../core/op_algo.h"
#include "../core/gradient_helper.h"

namespace mlfe{ 

class MaxPoolGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        using Ints = std::vector<type::int32::T>;
        VecTensor in_grads;
        Tensor x = y.get_children()[0];
        Tensor idx = y.get_context().get_attr<Tensor>("idx");
        Tensor dx = functional::create_variable(x.shape());
        auto y_ctx = y.get_context();
        OpAlgoContext ctx("MaxPoolGradient");

        dx.add_child(x);
        dx.add_child(y);
        dx.add_child(dy);
        
        ctx.add_attr({"kernel", y_ctx.get_attr<Ints>("kernel")});
        ctx.add_attr({"stride", y_ctx.get_attr<Ints>("stride")});
        ctx.add_attr({"padding", y_ctx.get_attr<Ints>("padding")});
        ctx.add_attr({"idx", idx});
        Tensor::AssignOpFunctor(dx, ctx);
        in_grads.push_back(dx);
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
    int out_h = (x.shape()[2] - kernel[0] + 2 * padding[0]) / stride[0] + 1;
    int out_w = (x.shape()[3] - kernel[1] + 2 * padding[1]) / stride[1] + 1;
    Tensor y = create_variable({x.shape()[0], x.shape()[1], out_h, out_w});
    Tensor idx = create_variable({x.shape()[0], x.shape()[1], out_h, out_w});
    OpAlgoContext ctx("MaxPool");
    y.add_child(x);
    ctx.add_attr({"kernel", kernel});
    ctx.add_attr({"stride", stride});
    ctx.add_attr({"padding", padding});
    ctx.add_attr({"idx", idx});
    Tensor::AssignOpFunctor(y, ctx);

    return y;
}

} // end namespace functional
} // end namespace mlfe
