#include "pool.h"
#include "../core/op_algo.h"
#include "../core/gradient_helper.h"

namespace mlfe{ 

// X Shape : N, C, H , W
// Y Shape : N, C, H', W'
REGIST_OP(MaxPool)
    .Input("X", "float32")
    .Output("IDX", "float32")
    .Output("Y", "float32")
    .Attr("filters_hw", "int32s")
    .Attr("strides", "int32s")
    .Attr("pads", "int32s")
    .ShapeInference([](OpDesignContext * odc){
        using IntVec = std::vector<type::int32::T>;
        auto x = odc->Input(0);
        auto idx = odc->Output(0);
        auto y = odc->Output(1);
        auto filters_hw = odc->GetAttr<IntVec>("filters_hw");
        auto strides = odc->GetAttr<IntVec>("strides");
        auto pads = odc->GetAttr<IntVec>("pads");
        auto x_shape = x.shape();
        int out_h = (x_shape[2] - filters_hw[0] + 2 * pads[0]) / strides[0] + 1;
        int out_w = (x_shape[3] - filters_hw[1] + 2 * pads[1]) / strides[1] + 1;
        idx.reshape({ x_shape[0], x_shape[1], out_h, out_w }, type::float32());
        y.reshape({ x_shape[0], x_shape[1], out_h, out_w }, type::float32());
    })
    .Finish();

REGIST_OP_GRAD(MaxPool)
    .Input("X", "float32")
    .Input("IDX", "float32")
    .Input("Y", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto dx = odc->Output(0);
        dx.reshape(x.shape(), type::float32());
    })
    .Finish();

// X Shape : N, C, H , W
// Y Shape : N, C, H', W'
REGIST_OP(AvgPool)
    .Input("X", "float32")
    .Output("IDX", "float32")
    .Output("Y", "float32")
    .Attr("filters_hw", "int32s")
    .Attr("strides", "int32s")
    .Attr("pads", "int32s")
    .ShapeInference([](OpDesignContext * odc){
        using IntVec = std::vector<type::int32::T>;
        auto x = odc->Input(0);
        auto idx = odc->Output(0);
        auto y = odc->Output(1);
        auto filters_hw = odc->GetAttr<IntVec>("filters_hw");
        auto strides = odc->GetAttr<IntVec>("strides");
        auto pads = odc->GetAttr<IntVec>("pads");
        auto x_shape = x.shape();
        int out_h = (x_shape[2] - filters_hw[0] + 2 * pads[0]) / strides[0] + 1;
        int out_w = (x_shape[3] - filters_hw[1] + 2 * pads[1]) / strides[1] + 1;
        idx.reshape({ x_shape[0], x_shape[1], out_h, out_w }, type::float32());
        y.reshape({ x_shape[0], x_shape[1], out_h, out_w }, type::float32());
    })
    .Finish();

REGIST_OP_GRAD(AvgPool)
    .Input("X", "float32")
    .Input("IDX", "float32")
    .Input("Y", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto dx = odc->Output(0);
        dx.reshape(x.shape(), type::float32());
    })
    .Finish();

class MaxPoolGradient : public GradientHelper{
public:
    MaxPoolGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

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
    AvgPoolGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

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
