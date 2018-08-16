#include "convolution.h"
#include "../core/op_dep.h"
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
        auto x = odc->Input("X");
        auto w = odc->Input("W");
        auto y = odc->Output("Y");
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
        auto x = odc->Input("X");
        auto w = odc->Input("W");
        auto dw = odc->Output("dW");
        auto dx = odc->Output("dX");
        dw.Reshape(w.Shape(), type::float32());
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class ConvolutionGradient : public GradientHelper{
public:
    ConvolutionGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    GradientHelper::HelperOut Get(Tensor dy) override{
        using IntVec = std::vector<type::int32::T>;
        Tensor x = odc->Input("X");
        Tensor w = odc->Input("W");
        Tensor y = odc->Output("Y");
        Tensor dw, dx;
        GradientHelper::GradientPairs pairs;

        auto dep = OpDependency::Builder("ConvolutionGradient")
            .Input(std::make_tuple("X", x))
            .Input(std::make_tuple("W", w))
            .Input(std::make_tuple("Y", y))
            .Input(std::make_tuple(Gradient("Y"), dy))
            .Output(std::make_tuple(Gradient("W"), dw))
            .Output(std::make_tuple(Gradient("X"), dx))
            .Attr({ "filters", odc->GetAttr<type::int32::T>("filters") })
            .Attr({ "filters_hw", odc->GetAttr<IntVec>("filters_hw") })
            .Attr({ "strides", odc->GetAttr<IntVec>("strides") })
            .Attr({ "pads", odc->GetAttr<IntVec>("pads") })
            .Finish();

        dx = Tensor::DependencyAdder(dep);
        pairs.push_back({ w, dw });

        return std::make_tuple(dx, pairs);
    }
};

REGIST_GRADIENT_HELPER(Convolution, ConvolutionGradient)

Tensor Conv(Tensor x,
            Tensor w,
            type::int32::T filters,
            std::vector<type::int32::T> filters_hw,
            std::vector<type::int32::T> strides,
            std::vector<type::int32::T> pads
            )
{
    Tensor y;

    auto dep = OpDependency::Builder("Convolution")
        .Input(std::make_tuple("X", x))
        .Input(std::make_tuple("W", w))
        .Output(std::make_tuple("Y", y))
        .Attr({ "filters", filters })
        .Attr({ "filters_hw", filters_hw })
        .Attr({ "strides", strides })
        .Attr({ "pads", pads })
        .Finish();

    y = Tensor::DependencyAdder(dep);

    return y;
}

} // end namespace functional
} // end namespace mlfe
