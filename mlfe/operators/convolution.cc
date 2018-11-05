#include "convolution.h"
#include "../core/op_algo.h"
#include "../core/gradient_helper.h"

namespace mlfe{ namespace functional{

// * Convolution Operator
//     - Input Shape = [batch, In Filters, Height, Width]
//     - Weight Shape = [Out Filters, In Filters, Weight_H, Weight_W]
//     - ConvOut Width = (Width + Padding * 2 - Weight_W) / Stride_W + 1
//     - ConvOut Height = (Height + Padding * 2 - Weight_H) / Stride_H + 1
//     - Output Shape = [batch, Out Filters, ConvOut Height, ConvOut Width]
REGIST_OP(Convolution)
    .Input("X", "float32")
    .Input("W", "float32")
    .Output("Y", "float32")
    .Attr("filters", "int32")
    .Attr("filters_hw", "int32s")
    .Attr("strides", "int32s")
    .Attr("pads", "int32s")
    .ShapeInference([](OpDesignContext * odc){
        using IntVec = std::vector<type::int32::T>;
        auto x = odc->Input(0);
        auto w = odc->Input(1);
        auto y = odc->Output(0);
        auto filters = odc->GetAttr<type::int32::T>("filters");
        auto filters_hw = odc->GetAttr<IntVec>("filters_hw");
        auto strides = odc->GetAttr<IntVec>("strides");
        auto pads = odc->GetAttr<IntVec>("pads");
        auto x_shape = x.Shape();
        w.Reshape({ filters, x_shape[1],
            filters_hw[0], filters_hw[1] }, type::float32());
        int out_h = (x_shape[2] - filters_hw[0] + 2 * pads[0]) / strides[0] + 1;
        int out_w = (x_shape[3] - filters_hw[1] + 2 * pads[1]) / strides[1] + 1;
        y.Reshape({ x.Shape()[0], filters, out_h, out_w }, type::float32());
    })
    .Finish();

// * Convolution Gradient Operator
//     - Input Shape = [batch, In Filters, Height, Width]
//     - Weight Shape = [Out Filters, In Filters, Weight_H, Weight_W]
//     - Output Gradient Shape = Output Shape
//     - Weight Gradient Shape = Weight Shape
//     - Input Gradient Shape = Input Shape
REGIST_OP_GRAD(Convolution)
    .Input("X", "float32")
    .Input("W", "float32")
    .Input("Y", "float32")
    .Input("dY", "float32")
    .Output("dW", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto w = odc->Input(1);
        auto dw = odc->Output(0);
        auto dx = odc->Output(1);
        dw.Reshape(w.Shape(), type::float32());
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class ConvolutionGradient : public GradientHelper{
public:
    ConvolutionGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y, 
                                Tensor dy
                               ) override{
        using IntVec = std::vector<type::int32::T>;
        TensorUmap gpair;
        Tensor x = y.get_children()[0];
        Tensor w = y.get_children()[1];
        Tensor dx = functional::variable(x.Shape());
        Tensor dw = functional::variable(w.Shape());
        OpAlgoContext ctx_x_grad("Conv2DGradientInputGradient");
        OpAlgoContext ctx_w_grad("Conv2DGradientFilterGradient");
        OpAlgoContext ctx = y.get_context();
        ctx_x_grad.add_attr({"strides", ctx.get_attr<IntVec>("strides")});
        ctx_x_grad.add_attr({"pads", ctx.get_attr<IntVec>("pads")});
        ctx_w_grad.add_attr({"strides", ctx.get_attr<IntVec>("strides")});
        ctx_w_grad.add_attr({"pads", ctx.get_attr<IntVec>("pads")});
        dx.add_child(w);
        dx.add_child(dy);
        dw.add_child(x);
        dw.add_child(dy);
        Tensor::AssignOpFunctor(dx, ctx_x_grad);
        Tensor::AssignOpFunctor(dw, ctx_w_grad);

        gpair[x] = dx;
        gpair[w] = dw;
        return gpair;
    }
};

REGIST_GRADIENT_HELPER(Convolution, ConvolutionGradient)

Tensor conv2d(Tensor x,
              Tensor w,
              std::vector<type::int32::T> strides,
              std::vector<type::int32::T> pads
              ){
    auto x_shape = x.Shape();
    auto w_shape = w.Shape();
    int out_h = (x_shape[2] - w_shape[2] + 2 * pads[0]) / strides[0] + 1;
    int out_w = (x_shape[3] - w_shape[3] + 2 * pads[1]) / strides[1] + 1;
    Tensor y = functional::variable({x_shape[0], w_shape[0], out_h, out_w});
    OpAlgoContext ctx("Convolution");
    ctx.add_attr({"strides", strides});
    ctx.add_attr({"pads", pads});
    y.add_child(x);
    y.add_child(w);
    Tensor::AssignOpFunctor(y, ctx);
    
    return y;
}

} // end namespace functional
} // end namespace mlfe
