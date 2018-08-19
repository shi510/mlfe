#include "pool.h"
#include "../core/op_dep.h"
#include "../core/gradient_helper.h"

namespace mlfe{ namespace functional{

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
        auto x = odc->Input("X");
        auto idx = odc->Output("IDX");
        auto y = odc->Output("Y");
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
        auto x = odc->Input("X");
        auto dx = odc->Output("dX");
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class MaxPoolGradient : public GradientHelper{
public:
    MaxPoolGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    GradientHelper::HelperOut Get(Tensor dy) override{
        using IntVec = std::vector<type::int32::T>;
        Tensor x = odc->Input("X");
        Tensor idx = odc->Output("IDX");
        Tensor y = odc->Output("Y");
        Tensor dx;
        GradientHelper::GradientPairs pairs;

        auto dep = OpDependency::Builder("MaxPoolGradient")
            .Input(std::make_tuple("X", x))
            .Input(std::make_tuple("IDX", idx))
            .Input(std::make_tuple("Y", y))
            .Input(std::make_tuple(Gradient("Y"), dy))
            .Output(std::make_tuple(Gradient("X"), dx))
            .Attr({ "filters_hw", odc->GetAttr<IntVec>("filters_hw") })
            .Attr({ "strides", odc->GetAttr<IntVec>("strides") })
            .Attr({ "pads", odc->GetAttr<IntVec>("pads") })
            .Finish();

        dx = Tensor::DependencyAdder(dep);

        return std::make_tuple(dx, pairs);
    }
};

REGIST_GRADIENT_HELPER(MaxPool, MaxPoolGradient)

Tensor MaxPool(Tensor x,
               std::vector<type::int32::T> filters_hw,
               std::vector<type::int32::T> strides,
               std::vector<type::int32::T> pads)
{
    Tensor idx, y;

    auto dep = OpDependency::Builder("MaxPool")
        .Input(std::make_tuple("X", x))
        .Output(std::make_tuple("IDX", idx))
        .Output(std::make_tuple("Y", y))
        .Attr({ "filters_hw", filters_hw })
        .Attr({ "strides", strides })
        .Attr({ "pads", pads })
        .Finish();

    y = Tensor::DependencyAdder(dep);

    return y;
}
} // end namespace functional
} // end namespace mlfe
