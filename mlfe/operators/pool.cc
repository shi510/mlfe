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
        auto x_shape = x.Shape();
        int out_h = (x_shape[2] - filters_hw[0] + 2 * pads[0]) / strides[0] + 1;
        int out_w = (x_shape[3] - filters_hw[1] + 2 * pads[1]) / strides[1] + 1;
        idx.Reshape({ x_shape[0], x_shape[1], out_h, out_w }, type::float32());
        y.Reshape({ x_shape[0], x_shape[1], out_h, out_w }, type::float32());
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
        dx.Reshape(x.Shape(), type::float32());
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
        auto x_shape = x.Shape();
        int out_h = (x_shape[2] - filters_hw[0] + 2 * pads[0]) / strides[0] + 1;
        int out_w = (x_shape[3] - filters_hw[1] + 2 * pads[1]) / strides[1] + 1;
        idx.Reshape({ x_shape[0], x_shape[1], out_h, out_w }, type::float32());
        y.Reshape({ x_shape[0], x_shape[1], out_h, out_w }, type::float32());
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
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class MaxPoolGradient : public GradientHelper{
public:
    MaxPoolGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y, 
                                Tensor dy
                               ) override{
        using Ints = std::vector<type::int32::T>;
        TensorUmap gpair;
        Tensor x = y.get_children()[0];
        Tensor idx = y.get_children()[1];
        Tensor dx = functional::variable(x.Shape());
        auto y_ctx = y.get_context();
        OpAlgoContext ctx("MaxPoolGradient");

        dx.add_child(x);
        dx.add_child(idx);
        dx.add_child(y);
        dx.add_child(dy);
        
        ctx.add_attr({"kernel", y_ctx.get_attr<Ints>("kernel")});
        ctx.add_attr({"stride", y_ctx.get_attr<Ints>("stride")});
        ctx.add_attr({"padding", y_ctx.get_attr<Ints>("padding")});
        Tensor::AssignOpFunctor(dx, ctx);

        gpair[x] = dx;
        return gpair;
    }
};

REGIST_GRADIENT_HELPER(MaxPool, MaxPoolGradient)

class AvgPoolGradient : public GradientHelper{
public:
    AvgPoolGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y, 
                                Tensor dy
                               ) override{
        using IntVec = std::vector<type::int32::T>;
        TensorUmap gpair;

        return gpair;
    }
};

REGIST_GRADIENT_HELPER(AvgPool, AvgPoolGradient)

namespace functional{

Tensor pool_max(Tensor x, 
                std::vector<int> kernel, 
                std::vector<int> stride, 
                std::vector<int> padding
               ){
    int out_h = (x.Shape()[2] - kernel[0] + 2 * padding[0]) / stride[0] + 1;
    int out_w = (x.Shape()[3] - kernel[1] + 2 * padding[1]) / stride[1] + 1;
    Tensor y = functional::variable({x.Shape()[0], x.Shape()[1], out_h, out_w});
    Tensor idx = functional::variable({x.Shape()[0], x.Shape()[1], out_h, out_w});
    OpAlgoContext ctx("MaxPool");
    y.add_child(x);
    y.add_child(idx);
    ctx.add_attr({"kernel", kernel});
    ctx.add_attr({"stride", stride});
    ctx.add_attr({"padding", padding});
    Tensor::AssignOpFunctor(y, ctx);

    return y;
}

} // end namespace functional
} // end namespace mlfe
