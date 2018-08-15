#include "reshape.h"
#include "../core/op_dep.h"
#include "../core/gradient_helper.h"

namespace mlfe{ namespace functional{

REGIST_OP(Reshape)
    .Input("X", "float32")
    .Output("Y", "float32")
    .Attr("shape", "int32s")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input("X");
        auto y = odc->Output("Y");
        auto shape = odc->GetAttr<std::vector<type::int32::T>>("shape");
        y.Reshape(shape, type::float32());
    })
    .Finish();

REGIST_OP_GRAD(Reshape)
    .Input("X", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input("X");
        auto dx = odc->Output("dX");
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class ReshapeGradient : public GradientHelper{
public:
    ReshapeGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    GradientHelper::HelperOut Get(Tensor dy) override{
        Tensor x = odc->Input("X");
        Tensor dx;
        GradientHelper::GradientPairs pairs;

        auto dep = OpDependency::Builder("ReshapeGradient")
            .Input({"X", x})
            .Input({ Gradient("Y"), dy })
            .Output({ Gradient("X"), dx })
            .Finish();

        dx = Tensor::DependencyAdder(dep);

        return{ dx, pairs };
    }
};

REGIST_GRADIENT_HELPER(Reshape, ReshapeGradient)

Tensor Reshape(Tensor x, std::vector<type::int32::T> shape){
    Tensor y;

    auto dep = OpDependency::Builder("Reshape")
        .Input({ "X", x })
        .Output({ "Y", y })
        .Attr({ "shape", shape })
        .Finish();

    y = Tensor::DependencyAdder(dep);

    return y;
}

} // end namespace functional
} // end namespace mlfe
