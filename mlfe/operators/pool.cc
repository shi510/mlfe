#include "pool.h"
#include "../core/op_algo.h"
#include "../core/gradient_helper.h"
#include "third_party/mkldnn/include/mkldnn.hpp"

namespace mlfe{ 

class MaxPoolGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        using Ints = std::vector<type::int32::T>;
        VecTensor in_grads;
        auto y_ctx = y.get_context();
        Tensor x = y_ctx.get_input(0);
        Tensor idx = y.get_context().get_attr<Tensor>("idx");
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
    int out_h = (x.shape()[2] - kernel[0] + 2 * padding[0]) / stride[0] + 1;
    int out_w = (x.shape()[3] - kernel[1] + 2 * padding[1]) / stride[1] + 1;
    Tensor y = create_variable({x.shape()[0], x.shape()[1], out_h, out_w});
    Tensor idx = create_variable({x.shape()[0], x.shape()[1], out_h, out_w});
    OpAlgoContext ctx("MaxPool");
    ctx.add_input(x);
    ctx.add_output(y);
    ctx.add_attr({"kernel", kernel});
    ctx.add_attr({"stride", stride});
    ctx.add_attr({"padding", padding});
    ctx.add_attr({"idx", idx});
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
